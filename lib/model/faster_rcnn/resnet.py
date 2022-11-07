from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': './data/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

import torchvision
class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers, pretrained, class_agnostic, unseen_classes, S_num, U_num):
    if num_layers == 101:
      self.model_func = torchvision.models.resnet101
      self.model_path = model_urls['resnet101']
    elif num_layers == 50:
      self.model_func = torchvision.models.resnet50
      self.model_path = model_urls['resnet50']
    else:
      assert False, "Not implemented."
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, unseen_classes, class_agnostic, S_num, U_num)

  def _init_modules(self):
    resnet = self.model_func()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %('./data/resnet101-5d3b4d8f.pth'))
      resnet.load_state_dict(torch.load('./data/resnet101-5d3b4d8f.pth'), strict=False)

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.S_num+1)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * (self.S_num+1))#self.n_classes)

    # Fix blocks
    # for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    # for p in self.RCNN_base[1].parameters(): p.requires_grad=False
    # for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    # self.RCNN_base.apply(set_bn_fix)
    # self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      # self.RCNN_base.eval()
      # self.RCNN_base[5].train()
      # self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      # self.RCNN_base.apply(set_bn_eval)
      # self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5)
    return fc7, fc7.mean(3).mean(2)
