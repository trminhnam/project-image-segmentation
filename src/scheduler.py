import torch
import torch.nn as nn


def apply_lambda_scheduler(optimizer, alpha=0.99):
    def lambda_rule(epoch):
        lr_l = alpha**epoch
        return lr_l

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler
