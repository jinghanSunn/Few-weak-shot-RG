from multiprocessing import pool
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
# from model.rpn.rpn import _RPN
from model.rpn.rpn import _RPN

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
import torchvision.ops as ops

from model.utils.net_utils import _crop_pool_layer, _affine_grid_gen, _affine_theta
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    
    sigma_2 = 1.0
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box  
    loss_box = loss_box.sum(1)
    loss_box = loss_box.mean()
    if not loss_box:
        # print(out_loss_box)
        pass
    return loss_box

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, unseen_classes, class_agnostic, S_num, U_num, Class_Semantic=False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.unseen_classes = unseen_classes
        self.S_num = S_num
        self.U_num = U_num
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer() #len(classes)) #n_class NO USE

        self.Class_Semantic = Class_Semantic
        if Class_Semantic == True:
            self.semantic_embedding = nn.Linear(2048, 768)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE

        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.CLS_loss = lambda x,y,z:F.cross_entropy(x, z)

        # for imprinting
        self.norm = True
        self.scale = True
        self.s = nn.Parameter(torch.FloatTensor([10])) 

    def forward(self, im_data, im_info, gt_boxes, num_boxes, return_pooled_feature=False, return_imp_feature=False, return_feature_and_loss=False, Class_Semantic_embedding=None):
        # print(im_data.shape)
        # print(gt_boxes.shape)
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # im_data.shape = Bx3xWxH
        base_feat = self.RCNN_base(im_data)

        # feed base feature map to RPN to obtain rois
        # base_feat.shape = Bx1024xwxh, im_info.shape = Bx3
        # gt_boxes.shape = Bx50x5, num_boxes.shape = B
        rois, rois_obj_score, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            # rois.shape = Bx2000x5
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # rois.shape = Bx128x5, rois_label.shape = Bx128, rois_target.shape = Bx128x4

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        # cfg.POOLING_SIZE = 7
        aligned_feat = ops.roi_align(base_feat, rois.view(-1, 5), [7, 7], spatial_scale=1./16)#, sampling_ratio=4)

        # feed pooled features to top model
        # pooled_feat.shape = B*128 x 1024 x 7 x 7
        layer4_feat, pooled_feat = self._head_to_tail(aligned_feat)
        # print(layer4_feat.shape) # [B*128, 2048, 4, 4]
        # pooled_feat.shape = B*128 x 2048
        if return_imp_feature:
            return pooled_feat
        

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        
        # bbox_pred.shape = 1*300 x 81*4, pooled_feat.shape = 1*300 x 2048
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels#
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        if self.norm:
            pooled_feat = self.l2_norm(pooled_feat)
        if self.scale:
            pooled_feat = self.s * pooled_feat
        cls_score = self.RCNN_cls_score(pooled_feat) # torch.Size([B*128, 34])  # self.RCNN_cls_score weight: nn.Linear(2048, self.S_num+1)
        # print(pooled_feat.shape)
        # print(cls_score.shape)
        if self.Class_Semantic:
            bbox_semantic_embedding = self.semantic_embedding(pooled_feat) # [b*128, 768]
            similarity = bbox_semantic_embedding*Class_Semantic_embedding # [b*128, 33] 
            similarity = F.softmax(similarity, 1)
            cls_score = F.softmax(cls_score,1)

            cls_score[:,1:] = 0.3*similarity + 0.7*cls_score[:,1:]
        

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # rois_label.max = 80, rois_label.min = 0

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws) 
            # print(RCNN_loss_bbox)

            # rpn_loss_cls = torch.Tensor([0]).cuda() + rpn_loss_cls
            # rpn_loss_bbox = torch.Tensor([0]).cuda() + rpn_loss_bbox
            # RCNN_loss_cls = torch.Tensor([0]).cuda() + RCNN_loss_cls
            # RCNN_loss_bbox = torch.Tensor([0]).cuda() + RCNN_loss_bbox

        cls_prob = F.softmax(cls_score, 1)
        


        if return_pooled_feature:
            att_prob = cls_prob.max(dim=1)[0] 
            att_avg_feats = pooled_feat * (att_prob.unsqueeze(1)) # [Case*128, 2048]
            
            # pooled_feat.shape,layer4_feat.shape torch.Size([6144, 2048]) torch.Size([6144, 2048, 4, 4])
            roi_size, feat_size, _, _ = layer4_feat.shape # [Case*128, 2048, 4, 4]
            patch_feats = layer4_feat.reshape(roi_size, feat_size, -1).permute(0, 2, 1) # [Case*128, 16, 2048]
            att_patch_feats = patch_feats * (att_prob.unsqueeze(1).unsqueeze(2))

            return att_patch_feats.mean(0), att_avg_feats.mean(0)
            


        
        if return_feature_and_loss:
            att_prob = cls_prob.max(dim=1)[0] 
            # print(pooled_feat.shape, att_prob.shape) # torch.Size([128, 2048]) torch.Size([1, 34])
            att_avg_feats = pooled_feat * (att_prob.unsqueeze(1)) # [Case*128, 2048]
            
            # pooled_feat.shape,layer4_feat.shape torch.Size([6144, 2048]) torch.Size([6144, 2048, 4, 4])
            roi_size, feat_size, _, _ = layer4_feat.shape # [Case*128, 2048, 4, 4]
            patch_feats = layer4_feat.reshape(roi_size, feat_size, -1).permute(0, 2, 1) # [Case*128, 16, 2048]
            att_patch_feats = patch_feats * (att_prob.unsqueeze(1).unsqueeze(2))
            
            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, att_patch_feats.mean(0), att_avg_feats.mean(0)
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.RCNN_cls_score.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.RCNN_cls_score.weight.data = w.div(norm.expand_as(w))
