import os
import torch
from torch.autograd import Function

def iou_coeff(output, target, type='sigmoid'):
    smooth = 1e-5
    if(type == 'sigmoid'):
        output = torch.sigmoid(output) 
        output[output <= 0.5] = 0
        output[output > 0.5] = 1     
    else:
        # output = torch.softmax(output, 1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        
    N = output.size(0)

    output_flat = output.view(N, -1).float()
    target_flat = target.view(N, -1).float()
    intersection = (output_flat * target_flat).sum(1)
    unionset = output_flat.sum(1) + target_flat.sum(1)
    loss = (intersection + smooth) / (unionset - intersection + smooth)

    return loss.sum() / N