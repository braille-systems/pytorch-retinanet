import torch
import torch.nn as nn

from .fpn import FPN50
from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_layers: torch.jit.Final[int]
    def __init__(self, num_layers =5, num_anchors=9, num_classes=20, num_fpn_layers=0, fpn_skip_layers=0, num_heads=1):
        '''
        :param num_layers: num output layers
        :param num_anchors:
        :param num_classes: can be int or list of ints (for several class labels
        :param num_fpn_layers: internal num of FPN layers = max(num_layers, num_fpn_layers)
        :param fpn_skip_layers: if != 0, these most grained output layers will bi skipped
        '''
        super(RetinaNet, self).__init__()
        self.fpn = FPN50(num_layers, num_fpn_layers, fpn_skip_layers)
        self.num_anchors = num_anchors
        self.total_num_classes = num_classes if isinstance(num_classes, int) else sum(num_classes)  # total class channels i.e. sum of all num_classes for all class groups
        self.loc_head = nn.ModuleList([self._make_head(self.num_anchors*4) for i in range(num_heads)])
        self.cls_head = nn.ModuleList([self._make_head(self.num_anchors*self.total_num_classes) for i in range(num_heads)])
        self.num_layers = num_layers

    def forward(self, x):
        fms = self.fpn(x)
        assert self.num_layers == len(fms)
        num_heads = len(self.loc_head)
        results: List[Tuple(Tensor, Tensor)] = []
        for i in range(num_heads):
            loc_preds: List[Tensor] = []
            cls_preds: List[Tensor] = []
            for fm in fms:
                loc_pred = self.loc_head[i](fm)
                cls_pred = self.cls_head[i](fm)
                loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
                cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.total_num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
                loc_preds.append(loc_pred)
                cls_preds.append(cls_pred)
            results.append((torch.cat(loc_preds,1), torch.cat(cls_preds,1)))
        if num_heads==1:
            return results[0]
        else:
            return results

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,224,224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()
