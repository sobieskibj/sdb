import torch

from .base import BaseMeanSystemResponse

import logging
log = logging.getLogger(__name__)

# clear existing handlers
if log.hasHandlers():
    log.handlers.clear()

# # add a StreamHandler to display output in the console or notebook
# stream_handler = logging.StreamHandler()

# # add the handler to the logger
# log.addHandler(stream_handler)


class CTProjectorMeanSystemResponse(BaseMeanSystemResponse):


    def __init__(self, path_assets: str, min_singular_value: float, use_idx_null: bool):
        self.use_idx_null = use_idx_null
        # Call the initializer of the parent class (nn.Module)
        super(CTProjectorMeanSystemResponse, self).__init__()
        U = torch.load(f"{path_assets}/U.pt")
        S = torch.load(f"{path_assets}/S.pt")
        V = torch.load(f"{path_assets}/V.pt")

        # register these as buffers
        self.register_buffer('U', U)
        self.register_buffer('S', S)
        self.register_buffer('V', V)
        self.register_buffer('invS', (1.0 / self.S).view(1, -1))

        # additional threshold based on condition number
        if self.use_idx_null: 
            self.invS[:, self.S < 1e-4 * self.S.max()] = 0.

        # used to mask out singular values below a given threshold
        self.min_singular_value = min_singular_value
        self.pick_idx = self.S >= min_singular_value
        log.info(f"Masking out {100 * (1 - (self.pick_idx.sum() / self.S.shape[0]))}% of singular values")

        # apply mask to U, S and V to save memory
        self.U = self.U[:, self.pick_idx]
        self.S = self.S[self.pick_idx]
        self.invS = self.invS[:, self.pick_idx]
        self.V = self.V[:, self.pick_idx]

        # Prevent the loaded matrices from being updated during training
        # This ensures these tensors are used only for forward pass computation, not for backpropagation
        # Make sure they are not trainable
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.invS.requires_grad = False
        self.V.requires_grad = False

        # Condition number for truncating singular values. Will be used to identify 'small' singular
        # values that can be ignored during pseudo-inverse reconstruction, as they are likely due to noise.
        # self.condition_number = 1e2

        # Identify the singular values that are not 'small' based on the condition number
        # The condition checks if the singular value 'S' is smaller than the maximum singular value divided by the condition number
        # This creates a boolean mask (self.idxNull) that can be used to filter out 'small' singular values
        # self.idxNull = self.S < torch.max(self.S) / self.condition_number
        
        # Find the index of the largest singular value that is still considered significant.
        # torch.where(~self.idxNull)[0] returns the indices of the non-zero elements in the boolean mask
        # torch.max() returns the maximum value in the tensor
        # index_min_singular_value = torch.max(torch.where(~self.idxNull)[0])

        # Create a list of equally spaced singular value indics, from 1 to the index of the 
        # largest valid singular value. This is used to restrict or select specific singular values
        # for pseudo-inverse reconstruction. torch.linspace generates 33 evenly spaced points from 0 to the
        # maximum index of the valid singular values. The first element is removed to avoid using 0 singular values.
        # self.singular_values_list = torch.linspace(0, index_min_singular_value, 33)[1:].to(torch.int32)

    
    def A(self, image):
        """
        Simulates the forward projection of the image to generate a sinogram (batch-based).
        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256).

        Returns:
            sinogram (torch.Tensor): The simulated sinogram tensor of shape (batch-based, 1, 72, 375).
            - 72: Number of angles for projection
            - 375: Projection length (based on geometry)
        """
        assert isinstance(image, torch.Tensor) # Ensure that the input image is a tensor
        assert len(image.shape) == 4  # Expecting batch, channel, height, width
        batch_size = image.shape[0]
        assert image.shape[1] == 1 # Validate that the input image tensor has 1 channel (grayscale)
        assert image.shape[2] == 256 # Validate that the input image tensor has a height of 256
        assert image.shape[3] == 256 # Validate that the input image tensor has a width of 256

        # Flatten image to 2D for projection
        # The input image is reshaped from (batch_size, 1, 256, 256) to (batch_size, 256*256).
        # This step prepares the image for matrix operations by converting the 2D spatial dimensions into a 1D vector
        x = image.view(batch_size, 256 * 256)

        # Compute VT_x = V^T * x
        # The transpose of the right singular vectors (V) is multiplied by the flattened image tensor (x).
        # This simulates applying a projection matrix (V) to the input image in the transformed space.
        # VT_x is reshaped to (batch_size, 256) to match the shape of the singular values (S).
        VT_x = torch.tensordot(x, self.V.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
        
        # Apply the singular values to the transformed image
        # Multiply each value in VT_x by the corresponding singular value in S.
        # This step simulates scaling the transformed data by the singular values to account for the 
        # magnitude of the projections
        S_VT_x = self.S.view(1, -1) * VT_x
        
        # Perform a tensor dot product between the scaled projections (S_VT_x) and the matrix U.
        # This simulates transforming the data back to the sinogram space.
        # The result is reshaped to (batch_size, 1, 72, 375) to match the expected sinogram shape.
        sinogram = torch.tensordot(S_VT_x, self.U, dims=([1], [1])).view(batch_size, 1, 72, 375)
        
        # Return the sinogram, which is the forward projeciton of the input image.
        return sinogram

    def AT(self, sinogram):
        """
        Inverse transformation from sinogram back to image space (batch-based).
        Args:
            sinogram (torch.Tensor): The input sinogram tensor of shape (batch-based, 1, 72, 375)

        Returns:
            AT_y (torch.Tensor): The reconstructed image tensor of shape (batch_size, 1, 256, 256)
        """
        assert isinstance(sinogram, torch.Tensor) # Ensure that the input sinogram is a PyTorch tensor
        assert len(sinogram.shape) == 4  # Expecting batch, channel, height, width
        batch_size = sinogram.shape[0]
        assert sinogram.shape[1] == 1
        assert sinogram.shape[2] == 72
        assert sinogram.shape[3] == 375

        # Flatten sinogram to 2D for back projection. The input sinogram tensor is reshaped from (batch_size, 1, 72, 375) to (batch_size, 72*375).
        # This prepares the sinogram for matrix operations by converting the 2D projection data into a 1D vector.
        y = sinogram.view(batch_size, 72 * 375)

        # Compute UT_y = U^T * y. The transpose of the left singular vectors (U) is multiplied by the flattened sinogram tensor (y).
        # This steps applies the inverse projection matrix (U^T), transforming the sinogram back towrad the original image space.
        # UT_y is reshaped to (batch_size, number of singular values) to match the shape of the singular values (S).
        UT_y = torch.tensordot(y, self.U.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
       
        # Apply the pseudo-inverse of the singular values to the transformed sinogram. 
        # Multiply each value in UT_y by the corresponding singular value in S.
        # This step scales the data in the transformed space by the singular values, simulating the inverse of the forward projection.
        S_UT_y = self.S.view(1, -1) * UT_y
        
        # Apply the final transformation (V matrix)
        # Perform a tensor dot product between the scaled inverse projections (S_UT_y) and the matrix V.
        # This step transforms the data from the singular value space back to the original image space.
        # The result is reshaped to (batch_size, 1, 256, 256) to match the original image shape.
        V_S_UT_y = torch.tensordot(S_UT_y, self.V, dims=([1], [1])).view(batch_size, 1, 256, 256)
        
        # The final reconstructed image is stored in AT_y and returned to the caller
        AT_y = V_S_UT_y
        return AT_y


    def pinvA(self, sinogram, singular_values=None):
        """
        Performs the pseudo-inverse reconstruction using a list of singular values (batch-based).
        Args:
            sinogram (torch.Tensor): The input sinogram tensor of shape (batch-based, 1, 72, 375)
            singular_values (list): A list of indices specifying ranges of singular values to use for the reconstruction.
        
        Returns:
            x_tilde_components (torch.Tensor): The reconstructed image components concatenated together based 
                on the pseudo-inverse reconstruction.  
        
        """
        assert isinstance(sinogram, torch.Tensor)
        assert len(sinogram.shape) == 4  # Expecting batch, channel, height, width
        batch_size = sinogram.shape[0]
        assert sinogram.shape[1] == 1
        assert sinogram.shape[2] == 72
        assert sinogram.shape[3] == 375

        # Flatten sinogram to 2D for reconstruction
        # The input sinogram tensor is reshaped from (batch_size, 1, 72, 375) to (batch_size, 72*375).
        # This prepares the sinogram for matrix operations by converting the 2D projection data into a 1D vector.
        y = sinogram.view(batch_size, 72 * 375)

        # x_tilde_components = []

        # # Handle the singular values and perform the reconstruction
        # # If no singular values are provided, the full set of singular values is used.
        # # This means that by default, all singular values are used for the reconstruction.
        # if singular_values is None:
        #     singular_values = [self.S.shape[0]]

        # # Loop over the provided range of singular values for reconstruction.
        # # This allows reconstruction based on different portions of the singular value spectrum.
        # for i in range(len(singular_values)):
        #     # Set the lower (sv_min) and upper (sv_max) boundary for the current range of singular values
        #     if i == 0:
        #         sv_min = 0
        #     else:
        #         sv_min = singular_values[i - 1]
        #     sv_max = singular_values[i]

        #     # Extract the U, S, and V matrices based on the current range of singular values
        #     _U = self.U[:, sv_min:sv_max]
        #     _S = self.S[sv_min:sv_max]
        #     _V = self.V[:, sv_min:sv_max]

        #     # Calculate the pseudo-inverse of the singular values. Identify very small singular values
        #     # and set their inverse to zeroto avoid division by near-zero values
        #     idxNull = _S < 1e-4 * torch.max(self.S) # Singular values below a threshold
        #     _invS = torch.zeros_like(_S)            # Initialize the inverse singular values
        #     _invS[~idxNull] = 1.0 / _S[~idxNull]    # Invert only the non-zero singular values

        #     # Compute the back-projected image component for the current singular value range.
        #     # Perform a tensor dot product between the flattened sinogram and the transpose of _U
        #     # This transforms the sinogram back toward the image space for the current singular vlaue range.
        #     UT_y = torch.tensordot(y, _U.T, dims=([1], [1])).view(batch_size, _S.shape[0])
            
        #     # Multiply by the pseudo-inverse singular values (_invS) to apply the inverse transformation
        #     S_UT_y = _invS.view(1, -1) * UT_y

        #     # Perform a tensor dot product between the transformed data and _V to project back to image space
        #     # This yields the reconstructed component for the current singular value range.
        #     V_S_UT_y = torch.tensordot(S_UT_y, _V, dims=([1], [1])).view(batch_size, 1, 256, 256)

        #     # Append the reconstructed component to the list of components. Each component corresponds
        #     # to a different range of singular values used for the reconstruction.
        #     x_tilde_components.append(V_S_UT_y)

        # # Concatenate all components along the channel dimension to form the full reconstructed image
        # x_tilde_components = torch.cat(x_tilde_components, dim=1)

        # return x_tilde_components

        UT_y = torch.tensordot(y, self.U.T, dims=([1], [1])).view(batch_size, self.S.shape[0])
        invS_UT_y = self.invS * UT_y
        pinvA_y = torch.tensordot(invS_UT_y, self.V, dims=([1], [1])).view(batch_size, 1, 256, 256)
        return pinvA_y
    

    def pinvATA(self, x):
        pass


    def reset_random_state(self, *args, **kwargs):
        pass


    def _HU_to_attenuation(self, image, scale_only):
        """
        scale_only == False:
        μ = (HU + 1000) * 0.1 / 1000 

        scale_only == True:
        μ = HU * 0.1 / 1000 
        """
        if scale_only:
            return (image) * 0.1 / 1000.0
        else:
            return (image + 1000.0) * 0.1 / 1000.0


    def _scaled_HU_to_attenuation(self, image):
        """
        Input images are scaled from HU to HU / 1000.
        To obtain HU range, we scale it back by 1000.
        """
        # map to real HU
        image = image * 1000.

        # map from HU to attenutation
        image = self._HU_to_attenuation(image, False)

        return image


    def inverse_hessian(self, image, meas_var=None, reg=None):
        """
        Compute the inverse Hessian of the image using the projector.
        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256)
            reg (float): Regularization parameter to stabilize the inverse Hessian computation
        
        Returns:
            inverse_hessian (torch.Tensor): The computed inverse Hessian of the input image
        """
        # Apply V^T to the image
        VT_x = torch.tensordot(image.view(image.shape[0], 256 * 256), self.V.T, dims=([1], [1])).view(image.shape[0], self.S.shape[0])

        if meas_var is None:
            meas_var = 1.0

        # Define the inverse of singular values squared
        if reg is None:
            invS2 = 1.0 / (self.S**2 / meas_var)
        else:
            invS2 = 1.0 / ((self.S**2 / meas_var) + reg)

        # Apply the inverse of singular values squared to the transformed image
        invS2_VT_x = invS2.view(1, -1) * VT_x

        # Apply V to the result
        V_invS2_VT_x = torch.tensordot(invS2_VT_x, self.V, dims=([1], [1])).view(image.shape[0], 1, 256, 256)

        return V_invS2_VT_x
    

    def null_space(self, image, singular_values=None):
        """
        Compute the null space of the image using V VT.
        Args:
            image (torch.Tensor): The input image tensor of shape (batch-based, 1, 256, 256)
            singular_values (list): A list of indices specifying ranges of singular values to use for the reconstruction.

        Returns:
            null_space (torch.Tensor): The computed null space of the input image
        """
        if singular_values is None:
            singular_values = [self.S.shape[0]]

        assert isinstance(singular_values, list)
        assert len(singular_values) ==1, 'for now only one singular value range is supported'
        sv_min = 0
        sv_max = singular_values[0]


        # print("DEBUG: self.V.device", self.V.device)
        # print("DEBUG: image.device", image.device)
        VT_x = torch.tensordot(image.view(image.shape[0], 256 * 256), self.V.T, dims=([1], [1])).view(image.shape[0], self.S.shape[0])
        range_space_transfer = torch.zeros_like(self.S)
        range_space_transfer[:sv_max] = 1.0 
        VT_x = VT_x * range_space_transfer
        range_space_x = torch.tensordot(VT_x, self.V, dims=([1], [1])).view(image.shape[0], 1, 256, 256)
        null_space_x = image - range_space_x

        return null_space_x
    