"""Ported from https://github.com/Hammour-steak/GOUB/blob/main/codes/models/lr_scheduler.py#L8"""
import torch
from collections import Counter, defaultdict
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLR_Restart(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma,
        gamma_,
        restarts,
        restart_weights,
        clear_state,
        last_epoch,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.gamma_ = gamma_
        self.clear_state = clear_state
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma_ ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]