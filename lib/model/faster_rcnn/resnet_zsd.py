from model.utils.config import cfg
from model.faster_rcnn.resnet import resnet
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import torchvision.ops as ops

def normal_init(m, mean, stddev, truncated=False):
  """
  weight initalizer: truncated normal and random normal.
  """
  if truncated:
      m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
  else:
      m.weight.data.normal_(mean, stddev)
      m.bias.data.zero_()

def cosine_similarity(x1, x2, eps=1e-8):
  # x1: B x emb_dim
  # x2: emb_dim x #cls
  dot12 = torch.mm(x1, x2) #-> B x #cls
  w1 = torch.norm(x1, 2, dim=1, keepdim=True) #-> B x 1
  w2 = torch.norm(x2, 2, dim=0, keepdim=True) #-> 1 x #cls
  w12 = torch.mm(w1, w2).clamp(min=eps) #-> B x #cls
  return (dot12/w12)

class ZSD_block(nn.Module):
  def __init__(self, embeddings_dim, num_target, context_top):
    super(ZSD_block, self).__init__()
    self.embeddings_dim = embeddings_dim
    self.num_target = num_target
    self.context_top = context_top
    self.cosine = False # True # wheather or not to use cosine classifier, may obtain higher pergformance with cosine classifier
 
    self.fc = nn.Linear(2048, self.embeddings_dim)
    normal_init(self.fc, 0, 0.01, cfg.TRAIN.TRUNCATED)

    self.fc_super_cls = nn.Linear(2048, num_target)
    normal_init(self.fc_super_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)

  def set_emb(self, cats, unseen_cats, embs, cls2asso, evaluate=False):
    embeddings = [[0]*self.embeddings_dim]
    all_cats = list(cats) + list(unseen_cats)
    assert all_cats[0] == '__background__'
    for c in all_cats:
      if c == '__background__':
        continue
      embeddings.append(embs[c])
    embeddings = torch.tensor(embeddings)

    # embedding for background: [1 0 0 0 ...]
    embeddings[0][0] = 1
    # l2-normalization for embeddings
    embeddings = embeddings/torch.norm(embeddings, dim=1, keepdim=True)

    # get distribution
    self.no_relation_cls = []
    id2asso_dist = embeddings.new_zeros([embeddings.size(0), self.num_target])
    for i,c in enumerate(all_cats):
      if c == '__background__':
        continue
      association = cls2asso[c]
      if type(association) is list: # seen class, assign a association distribution with unseen targets
        if sum(association) == 0:
            self.no_relation_cls.append([i, c])
        id2asso_dist[i] = torch.tensor(association)
      else: # unseen targets, assign 1 to their indices
        id2asso_dist[i][association] = 1
    if self.no_relation_cls: # in case some seen classes have no concept association to the unseen targets
    	print(self.no_relation_cls)

    # embs are with background as the 0st element
    if evaluate:
        embeddings_cats = torch.transpose(embeddings, 0, 1)
    else:
        embeddings_cats = torch.transpose(embeddings[:len(cats)], 0, 1)
        id2asso_dist = id2asso_dist[:len(cats)]
  
    self.embeddings = []
    self.id2asso_dists = []
    for i in range(torch.cuda.device_count()):
      self.embeddings.append(embeddings_cats.cuda(device=i))
      self.id2asso_dists.append(id2asso_dist.cuda(device=i))

  def forward(self, x, aligned_feat, rois):
    emb = self.embeddings[x.get_device()]
    x_emb = self.fc(x)
    if self.cosine:
        x_emb = 20*cosine_similarity(x_emb, emb)
    else:
        x_emb = torch.mm(x_emb, emb)

    x_context = self.context_top(aligned_feat).mean(3).mean(2)
    x_asso = self.fc_super_cls(x_context)
    return x_emb, x_asso

  def get_distribution(self, cats):
    id2asso_dist = self.id2asso_dists[cats.get_device()]
    return id2asso_dist[cats]
  
  def asso2cls(self, asso_prob):
    id2asso_dist = self.id2asso_dists[asso_prob.get_device()]
    return torch.transpose(id2asso_dist, 0, 1)

  def asso_loss(self, asso_score, cls_label):
    asso_dist = self.get_distribution(cls_label)
    loss = F.binary_cross_entropy(torch.sigmoid(asso_score), asso_dist)
    return loss


class resnet_zsd(resnet):
    def __init__(self, classes, num_layers=101, pretrained=True, class_agnostic=True, unseen_classes=[], S_num=80, U_num=40):
        resnet.__init__(self, classes, num_layers, pretrained, class_agnostic, unseen_classes, S_num, U_num)

    def zsd(self, embeddings_dim, num_target):
        eval_blocks = [
            self.RCNN_base,
            self.RCNN_rpn,
            self.RCNN_top,
            self.RCNN_bbox_pred,
            self.RCNN_cls_score
        ]
        context_top = copy.deepcopy(self.RCNN_top)
        
        for block in eval_blocks:
            block.eval()
            for p in block.parameters(): p.requires_grad=False

        self.cls_score = ZSD_block(embeddings_dim, num_target, context_top)
        self.ASSO_loss = self.cls_score.asso_loss

    def set_emb(self, cats, unseen_cats, embeddings, cls2asso, evaluate=False):
        self.cls_score.set_emb(cats, unseen_cats, embeddings, cls2asso, evaluate)

    def train(self, mode=True):
        self.training = True
        train_blocks = [self.cls_score]
        for block in train_blocks:
            block.train()

    def eval(self):
        self.training = False
        for module in self.children():
            module.eval()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, mode="train"):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        base_feat = self.RCNN_base(im_data)
        rois, rois_obj_score, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

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

        aligned_feat = ops.roi_align(base_feat, rois.view(-1, 5), [7, 7], spatial_scale=1./16)#, sampling_ratio=4)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(aligned_feat) # resnet layer4

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability and concept association
        emb_score, asso_score = self.cls_score(pooled_feat, aligned_feat, rois)

        asso_prob = torch.sigmoid(asso_score)
        asso_prob = torch.mm(asso_prob, self.cls_score.asso2cls(asso_prob))

        RCNN_loss_cls = 0
        RCNN_loss_asso = 0
        if self.training:
            RCNN_loss_cls = F.cross_entropy(emb_score, rois_label)
            RCNN_loss_asso = self.ASSO_loss(asso_score, rois_label)

        cls_prob = F.softmax(emb_score, 1)
        if mode == "zsd" or mode == "ori_zsd":
            cls_prob = torch.cat([cls_prob[:, 0:1], cls_prob[:, -self.U_num:]], dim=-1)
            asso_prob = torch.cat([asso_prob[:, 0:1], asso_prob[:, -self.U_num:]], dim=-1)
        
        if mode == "zsd" or mode == "gzsd":
        	# in case some seen classes have no concept association to the unseen targets
            if self.cls_score.no_relation_cls != [] and mode == "gzsd":
                for i, c in self.cls_score.no_relation_cls:
                    asso_prob[:, i] = 1
            cls_prob = cls_prob * asso_prob

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        return rois, cls_prob, bbox_pred, RCNN_loss_cls, RCNN_loss_asso, rois_label