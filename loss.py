from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_dict = {'loc':0, 'cls':0}

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data, 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t)  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w.data, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25

        t = one_hot_embedding(y.data, 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t)

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def focal_loss_3(self, inputs, targets):
        '''Focal loss alternative 3

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        gamma = 2
        targets = targets.unsqueeze(1).float()
        #GVNC 1) only for 1 class 2) alpha is missed
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        f_loss = (1 - pt) ** gamma * bce_loss
        return f_loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        del pos
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        del mask
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        del masked_loc_preds
        del masked_loc_targets

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        num_neg = pos_neg.data.long().sum() * cls_preds.shape[-1]
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        del mask
        masked_cls_targets = cls_targets[pos_neg]
        del pos_neg
        cls_loss = self.focal_loss(masked_cls_preds, masked_cls_targets)
        del masked_cls_preds
        del masked_cls_targets

        num_pos = max(1, num_pos)
        num_neg = max(1, num_neg)
        #print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss/num_pos, cls_loss/num_neg), end=' | ')
        loss = loc_loss/num_pos+cls_loss/num_neg
        self.loss_dict = {'loss':loss, 'loc':loc_loss/num_pos, 'cls':cls_loss/num_neg}
        return loss
