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

def attack(dataloader, model, device, class_weights, num_steps, step_size, eps, clamp=(0, 1)):
    images = dataloader.dataset.__getimages__()
    labels = dataloader.dataset.__getlabels__()

    labels = torch.argmax(labels, dim=1)

    images = images.to(device)
    labels = labels.to(device)

    print(labels.shape)

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    adversarial_images = []

    criterion = FocalLoss(class_weights.to(device))

    for i, [input, label] in enumerate(dataloader):

        x_adv = input.clone().detach().requires_grad_(True).to(device)
        num_channels = input.shape[1]

        for i in range(num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            output = model(_x_adv)

            loss = criterion(output, label)
            loss.backward()

            with torch.no_grad():
                if step_norm == 'inf':
                    gradients = _x_adv.grad.sign() * step_size
                else:
                    # Note .view() assumes batched image data as 4D tensor
                    gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                        .norm(step_norm, dim=-1)\
                        .view(-1, num_channels, 1, 1)
            
                x_adv += gradients

            x_adv = torch.max(torch.min(x_adv, input + eps), input - eps)
            
            # delta = x_adv - input

            # mask = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1) <= eps
            
            # scaling_factor = delta.view(delta.shape[0], -1).norm(eps_norm, dim=1)
            # scaling_factor[mask] = eps

            # delta *= eps / scaling_factor.view(-1, 1, 1, 1)

            # x_adv = x + delta

            x_adv = x_adv.clamp(*clamp)

        adversarial_images.append(x_adv)

    return adversarial_images

