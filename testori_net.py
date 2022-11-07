from email.policy import strict
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import ipdb
import time
import json
from tqdm import tqdm

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
from lib.modules.datasets import FFAIRDataset
from lib.modules.tokenizers import Tokenizer
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections

from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='ffair', type=str)
  parser.add_argument('--net', dest='net',
                      help='res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="./DPIF-FFAIR-main-output/models/",
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
                      default=10, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=255, type=int)
  parser.add_argument('--note', type=str, default='', help='note for model path')
  parser.add_argument('--num_classes',
                    default=34, type=int)
  # Data input settings
  parser.add_argument('--image_dir', type=str, default='./data/FFAIRNew/FFA-IR_image/EYE/', help='the path to the images.')
  parser.add_argument('--ann_path', type=str, default='./data/FFAIRNew/updated_ffair_annotation.csv', help='the path to the annotation file.')
  parser.add_argument('--box_ann_path', type=str, default='./data/FFAIRNew/lesion_info_new.json', help='the path to the bbox annotation file.')

  # Data loader settings
  parser.add_argument('--dataset_name', type=str, default='ffair', help='the dataset to be used.')
  parser.add_argument('--max_seq_length', type=int, default=90, help='the maximum sequence length of the reports.')
  parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.') # ???????

  parser.add_argument('--split', type=str, default='train', help='note for model path')

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()
  args.class_agnostic = True # False

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "coco6515":
      args.imdb_name = "coco6515_2014_train"
      args.imdbval_name = "coco6515_2014_test"
      cats_path = "data/categories_coco_65_15.json"
      S_num = 65
      U_num = 15
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "voc164":
      args.imdb_name = "voc164_0712_train"
      args.imdbval_name = "voc164_2007_test"
      cats_path = "data/categories_voc_16_4.json"
      S_num = 16
      U_num = 4
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "vg478130":
      args.imdb_name = "vg478130_2017_train"
      args.imdbval_name = "vg478130_2017_test"
      cats_path = "data/categories_vg_478_130.json"
      S_num = 478
      U_num = 130
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "ffair":
      args.imdb_name = "ffair_train"
      args.imdbval_name = "ffair_test"
      # cats_path = "data/categories_coco_65_15.json"
      S_num = 33
      U_num = 7
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  # all_cats = json.load(open(cats_path))
  # novel_classes = all_cats['unseen']
  # imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, training=False, novel_classes=novel_classes)
  # imdb.competition_mode(on=True)

  # print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + args.note + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn{}_{}_{}_{}.pth'.format(args.note, args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'res101':
    num_layer = 101
  elif args.net == 'res50':
    num_layer = 50
  else:
    print("network is not defined")
    exit()
  
  if args.split == 'val' or args.split == 'test':
        fasterRCNN = resnet(U_num, num_layer, False, args.class_agnostic, None, U_num, U_num)
  else:
    fasterRCNN = resnet(args.num_classes, num_layer, False, args.class_agnostic, None, S_num, U_num)
  fasterRCNN.create_architecture()
  num_classes = args.num_classes

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  for k in list(checkpoint['model'].keys()):
    if (k.find('_cls_score')==-1 and k.find('cls_score')!=-1):
      print(k)
      del checkpoint['model'][k]
  fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100
  thresh = 0.000001

  save_name = 'faster_rcnn_{}'.format(args.checksession)
  

  output_dir = get_output_dir()
  # dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        # num_classes, training=False, normalize = False)
  tokenizer = Tokenizer(args)
  dataset = FFAIRDataset(args, tokenizer, args.split, return_path=True)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)
  num_images = len(dataloader)
  all_boxes = [[[] for _ in range(30000)]
               for _ in range(num_classes)]
  all_image_ids = [[[] for _ in range(30000)]
               for _ in range(num_classes)]
  # print(np.array(all_image_ids).shape)
  # data_iter = iter(dataloader)
  
  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections_%d.pkl'%args.checkepoch)
  
  if True: #False: # wheather or not to use the cached data to evaluate
    with torch.no_grad():
      fasterRCNN.eval()
      empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
      # for i in range(num_images):
          # data = next(data_iter)
      i = 0
      all_image_path = []
      for data in tqdm(dataloader):
        im_data_all, im_info_all, gt_boxes_all, num_boxes_all, image_paths = data # [1,b,...]
        all_image_path.extend(image_paths)
        if args.cuda:
          im_data_batch = im_data_all[0].cuda() # [b,...]
          im_info_batch = im_info_all.cuda() # [1, 3]
          num_boxes_batch = num_boxes_all[0].cuda()
          gt_boxes_batch = gt_boxes_all[0].cuda()
          # print("im_info_all",im_info_all.shape)
          # print(len(image_paths))
        for k in range(len(im_data_batch)):
          im_data = im_data_batch[k].unsqueeze(0)
          im_info = im_info_batch # [1, 3]
          gt_boxes = gt_boxes_batch[k].unsqueeze(0)
          num_boxes = num_boxes_batch[k].unsqueeze(0)
          image_id = image_paths[k]
          
          det_tic = time.time()

          # print("im_data.shape",im_data.shape) # im_data.shape torch.Size([1, 3, 224, 224])
          # print("gt_boxes.shape",gt_boxes.shape) # [1, 8, 5]

          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

          scores = cls_prob.data
          boxes = rois.data[:, :, 1:5]

          if cfg.TEST.BBOX_REG:#False:#
              # Apply bounding-box regression deltas
              box_deltas = bbox_pred.data
              if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
              # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                              + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                              + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * num_classes)
              # print(boxes.shape) # [37, 300, 4]
              # print(box_deltas.shape) # [1, 11100, 4]
              pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
              pred_boxes = clip_boxes(pred_boxes, im_info.data, 1) # box最大值只有223
          else:
              # Simply repeat the boxes, once for each class
              _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
              pred_boxes = _.cuda() if args.cuda > 0 else _
          # print("pred box", pred_boxes)
          # print("data[1][0][2].item()", data[1][0][2].item() )
          # ipdb.set_trace()
          pred_boxes /= data[1][0][2].item() # resize to origin image shape

          # print("pred_boxes",pred_boxes)
          # print("ground truth", gt_boxes)

          scores = scores.squeeze()
          # print("scores.shape", scores.shape) # [37, 300, 34]
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
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = ops.nms(cls_dets[:,:4], cls_dets[:,4], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                all_boxes[j][i] = cls_dets.cpu().numpy()
              else:
                all_boxes[j][i] = empty_array
              all_image_ids[j][i] = image_id
          # Limit to max_per_image detections *over all classes*
          if max_per_image > 0:
              image_scores = np.hstack([all_boxes[j][i][:, -1]
                                        for j in range(beg, num_classes)])
              if len(image_scores) > max_per_image:
                  image_thresh = np.sort(image_scores)[-max_per_image]
                  for j in range(beg, num_classes):
                      keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                      all_boxes[j][i] = all_boxes[j][i][keep, :]
                      all_image_ids[j][i] = image_id
          misc_toc = time.time()
          nms_time = misc_toc - misc_tic

          sys.stdout.write('im_detect_images: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
              .format(k + 1, len(im_data_batch), detect_time, nms_time))
          i+=1
          # sys.stdout.flush()
        # sys.stdout.write('im_detect_cases: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
        #     .format(i + 1, num_images, detect_time, nms_time))
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
  else:
    with open(det_file, 'rb') as f:
        all_boxes = pickle.load(f)

  print('Evaluating detections')
  # print("main all_image_ids",np.array(all_image_ids).shape)
  dataset.evaluate_detections(all_boxes, output_dir, all_image_ids)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
