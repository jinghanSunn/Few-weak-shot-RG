from random import shuffle
from cv2 import imdecode
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

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from lib.modules.tokenizers import Tokenizer

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from lib.modules.datasets_imprinted import FFAIRDataset
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.resnet import resnet

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
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=1000, type=int)
  parser.add_argument('--num_class',
                      default=33, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)
  # Data input settings
  parser.add_argument('--image_dir', type=str, default='./data/FFAIRNew/FFA-IR_image/EYE/', help='the path to the images.')
  parser.add_argument('--ann_path', type=str, default='./data/FFAIRNew/updated_ffair_annotation.csv', help='the path to the annotation file.')
  parser.add_argument('--box_ann_path', type=str, default='./data/FFAIRNew/lesion_info_new.json', help='the path to the bbox annotation file.')

  # Data loader settings
  parser.add_argument('--dataset_name', type=str, default='ffair', help='the dataset to be used.')
  parser.add_argument('--max_seq_length', type=int, default=90, help='the maximum sequence length of the reports.')
  parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.') # ???????
  # parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
  # parser.add_argument('--batch_size', type=int, default=2, help='the number of samples for a batch')


  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="./DPIF-FFAIR-main-output/models",
                      type=str)
  parser.add_argument('--note', type=str, default='', help='note for model path')
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.0001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--resume',
                      help='resume checkpoint or not',
                      default='', type=str)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
  parser.add_argument('--split', type=str, default='test', help='note for model path')
  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_whole_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_whole_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_whole_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_whole_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()
  args.class_agnostic = True # False

  print('Called with args:')
  print(args)

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
      S_num = 479
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
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  # novel_classes = json.load(open(cats_path))['unseen']
  # imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, novel_classes=novel_classes)
  # train_size = len(roidb)

  # print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + args.note + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # sampler_batch = sampler(train_size, args.batch_size)

  # dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                          #  imdb.num_classes, training=True)
  tokenizer = Tokenizer(args) # to check ???????????????
  dataset = FFAIRDataset(args, tokenizer, 'val')

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'res101':
    num_layer = 101
  elif args.net == 'res50':
    num_layer = 50
  else:
    print("network is not defined")
    exit()

  fasterRCNN = resnet(args.num_class, num_layer, True, args.class_agnostic, None, S_num, U_num)
  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr

  if args.resume!='':
    # load_name = os.path.join(output_dir,
      # 'faster_rcnn_{}_{}_{}.pth'.format( args.checksession, args.checkepoch, args.checkpoint))
    load_name = args.resume
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = 0
    fasterRCNN.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % (load_name))

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
  optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  # iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  # for epoch in range(args.start_epoch, 1):
  # setting to train mode
  fasterRCNN.eval()
  loss_temp = 0
  step = 0
  start = time.time()

  ############################################## imprinted weight #############################################################
  new_weight_dict = dict()
  new_weight = torch.zeros(7+1, 2048)
  record_len = torch.zeros(7+1)
  with torch.no_grad():
    for data in dataloader:
      im_data, im_info, gt_boxes, num_boxes = data
      im_data = im_data.squeeze()
      # gt_boxes = gt_boxes.squeeze(0)
      gt_boxes = gt_boxes[0]
      num_boxes = num_boxes.squeeze()
      

      index = np.array(num_boxes!=0).flatten()
      im_data = im_data[index]
      gt_boxes = gt_boxes[index] # [x, 1, 5]
      num_boxes = num_boxes[index]

      if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

      # get class
      cls = gt_boxes[0][0][4].item()

      # fasterRCNN.zero_grad()
      feature = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,return_imp_feature=True) 
      if cls!=0:
        # new_weight[cls] += feature
        if cls not in new_weight_dict.keys():
          new_weight_dict[cls] = feature
        else:
          new_weight_dict[cls] = torch.cat([new_weight_dict[cls], feature], dim=0)
    
      loss_temp = 0
      start = time.time()
      sys.stdout.flush()
    step += 1

  for cls in new_weight_dict.keys():
        cls_feature = new_weight_dict[cls].mean(0)
        cls_feature = cls_feature / cls_feature.norm(p=2)
        new_weight[int(cls)] = cls_feature
  print(new_weight[1:].shape)

  print(fasterRCNN.RCNN_cls_score.weight.data[0].unsqueeze(0).shape)
  weight = torch.cat((fasterRCNN.RCNN_cls_score.weight.data[0].unsqueeze(0), new_weight[1:].cuda()),dim=0) # 背景的weight不变
  fasterRCNN.RCNN_cls_score = nn.Linear(2048, U_num+1, bias=False)
  fasterRCNN.RCNN_cls_score.weight.data = weight

  ############################################## finetune #############################################################
  fasterRCNN.train()
  for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    step = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    for data in dataloader:
      im_data, im_info, gt_boxes, num_boxes = data
      im_data = im_data.squeeze()
      gt_boxes = gt_boxes[0]
      num_boxes = num_boxes.squeeze()
      
      index = np.array(num_boxes!=0).flatten()
      im_data = im_data[index]
      gt_boxes = gt_boxes[index] # [x, 1, 5]
      num_boxes = num_boxes[index]

      if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
          + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.detach().item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      fasterRCNN.weight_norm()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, len(dataloader), loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

      
        loss_temp = 0
        start = time.time()
        sys.stdout.flush()
      step += 1


    save_name = os.path.join(output_dir, 'faster_rcnn{}_{}_{}_{}.pth'.format(args.note, args.session, args.checkepoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': args.checkepoch,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
