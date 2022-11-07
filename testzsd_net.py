import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
import torchvision.ops as ops
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections

from model.faster_rcnn.resnet_zsd import resnet_zsd as resnet


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                      help='res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)

  parser.add_argument('--gzsd', dest='gzsd',
                      help='whether GZSD',
                      action='store_true')

  parser.add_argument('--noasso', dest='noasso',
                      help='whether or not to use association',
                      action='store_true')

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()
  args.class_agnostic = True

  noasso = args.noasso
  GZSD = args.gzsd

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "coco6515":
      args.imdb_name = "coco6515_2014_train"
      args.imdbval_name = "coco6515_2014_test"
      cats_path = "data/categories_coco_65_15.json"
      cls2asso_distribution_path = "./data/cls2asso_coco_w2v_dist_5.json"
      S_num = 65
      U_num = 15
      num_target = 15
      embeddings_dim = 300
      embeddings_file = "./data/embeddings_w2v.json"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "voc164":
      args.imdb_name = "voc164_0712_train"
      args.imdbval_name = "voc164_2007_test"
      cats_path = "data/categories_voc_16_4.json"
      cls2asso_distribution_path = "./data/cls2asso_voc_attr_dist_2.json"
      S_num = 16
      U_num = 4
      num_target = 4
      embeddings_dim = 64
      embeddings_file = "./data/voc_attr_all.json"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "vg478130":
      args.imdb_name = "vg478130_2017_train"
      args.imdbval_name = "vg478130_2017_test"
      cats_path = "data/categories_vg_478_130.json"
      cls2asso_distribution_path = "./data/cls2asso_vg_w2v_dist_5.json"
      S_num = 478
      U_num = 130
      num_target = 130
      embeddings_dim = 300
      embeddings_file = "./data/embeddings_vg_w2v.json"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  all_cats = json.load(open(cats_path))
  if GZSD:
    ignore_classes = []
    target_classes = ['__background__'] + all_cats['seen'] + all_cats['unseen']
  else:
    ignore_classes = all_cats['seen']
    target_classes = ['__background__'] + all_cats['unseen']
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, training=False, novel_classes=ignore_classes)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint)) 
  num_classes = len(target_classes)

  if args.net == 'res101':
    num_layer = 101
  elif args.net == 'res50':
    num_layer = 50
  else:
    print("network is not defined")
    exit()
  
  fasterRCNN = resnet(imdb.classes, num_layer, False, args.class_agnostic, None, S_num, U_num)
  fasterRCNN.create_architecture()

  fasterRCNN.zsd(embeddings_dim, num_target)
  with open(embeddings_file) as f:
    embeddings = json.load(f)
  with open(cls2asso_distribution_path) as f:
    cls2asso = json.load(f)
  fasterRCNN.set_emb(['__background__']+all_cats["seen"], all_cats["unseen"], embeddings, cls2asso, evaluate=True)
  print("loaded embeddings %d/%d"%(len(imdb.classes)-1,len(embeddings)))
  print("%s, WORD_EMB:w2v, %s"%("GZSD" if GZSD else "ZSD", "noasso" if noasso else "association:multiple"))
  print("loaded embeddings from %s & association from %s"%(embeddings_file, cls2asso_distribution_path))


  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  state_dict = checkpoint['model']

  fasterRCNN.load_state_dict(state_dict)
  print('load model successfully!')

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100
  thresh = 0.000001

  save_name = 'faster_rcnn_{}'.format(args.checksession)
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in range(num_images)]
               for _ in range(num_classes)]

  output_dir = get_output_dir(imdb, save_name)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections_%s_%d%s.pkl'%(
    "gzsd" if GZSD else "zsd", args.checkepoch, "_noasso" if noasso else "")
  )
  
  if True: #False: # wheather or not to use the cached data to evaluate
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    for i in range(num_images):
        data = next(data_iter)
        im_data, im_info, gt_boxes, num_boxes = data
        if args.cuda:
          im_data = im_data.cuda()
          im_info = im_info.cuda()
          num_boxes = num_boxes.cuda()
          gt_boxes = gt_boxes.cuda()

        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        RCNN_loss_cls, RCNN_loss_asso, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, mode="%s%szsd"%(
          "ori_" if noasso else "", "g" if GZSD else "")
        )
        
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:#False:#
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              # if args.class_agnostic:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas = box_deltas.view(1, -1, 4)
              # else:
              #     box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
              #                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              #     box_deltas = box_deltas.view(1, -1, 4 * num_classes)

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
            pred_boxes = _.cuda() if args.cuda > 0 else _

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        
        beg = 1
        for j in range(beg, num_classes):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              # if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
              # else:
              #   cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              cls_dets = cls_dets[order]
              keep = ops.nms(cls_dets[:,:4], cls_dets[:,4], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
              all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(beg, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(beg, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  else:
    with open(det_file, 'rb') as f:
        all_boxes = pickle.load(f)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir, target_classes, U_classes=all_cats['unseen'])

  end = time.time() 
  print("test time: %0.4fs" % (end - start))
