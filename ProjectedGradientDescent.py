import os
import torch
import numpy as np
from utils import save_image, make_dirs

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        ce_loss = nn.BCELoss()(probs, labels)
        # print(type(probs), probs, self.gamma)
        weight = (1 - probs).pow(self.gamma)
        loss = ce_loss  # Initialize loss with cross-entropy loss
        if self.class_weights is not None:
            weight = weight * self.class_weights
            loss = loss * weight
        return loss
    

def pgd_attack(dataloader, model, device, class_weights, eps=0.3, alpha=2/255, iters=40):
    outputs = list()

    for i, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        loss = FocalLoss(class_weights)
        ori_image = torch.clone(image)

        for j in range(iters) :    
            image.requires_grad = True
            output = model(image)

            model.zero_grad()
            cost = loss(output.float(), label.float()).to(device)
            cost.backward()

            adv_image = image + alpha*image.grad.sign()
            eta = torch.clamp(adv_image - ori_image, min=-eps, max=eps)
            image = torch.clamp(ori_image + eta, min=0, max=1).detach_()
        outputs.append(image)
    return outputs