# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 20:56
# @Author  : uhauha2929
from typing import List

import torch
import torch.nn.functional as F


def focal_loss(inputs: torch.Tensor,
               targets: torch.Tensor,
               decay: float = 2,
               weight: List[float] = None,
               device: torch.device = None):
    inputs = F.softmax(inputs, -1)
    if weight is not None:
        weight = torch.Tensor([weight[i] for i in targets]).to(device)
    targets = targets.view(-1, 1)
    inputs = torch.gather(inputs, 1, targets)
    neg_inputs = 1 - inputs
    cross_entropy = -torch.log(inputs)
    loss = torch.pow(neg_inputs, decay) * cross_entropy
    if weight is not None:
        loss = weight * loss
    return loss.sum()
