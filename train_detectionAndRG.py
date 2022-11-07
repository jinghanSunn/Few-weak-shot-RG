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
import torchvision.ops as ops
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from lib.modules.tokenizers import Tokenizer

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from lib.modules.datasets import FFAIRDataset
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from lib.model.ffair_for_rpn import FFAIRModel
from modules.trainer import Trainer
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.loss import compute_loss

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

  # Optimization
  parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
  parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
  parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
  parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
  parser.add_argument('--amsgrad', type=bool, default=True, help='.')

  # Learning Rate Scheduler
  parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
  parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
  parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

  # Model settings (for visual extractor)
  parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
  parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

  # Model settings (for Transformer)
  parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
  parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
  parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
  parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
  parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
  parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
  parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
  parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
  parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
  parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
  parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
  parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

  parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
  parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
  parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
  parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
  parser.add_argument('--monitor_metric', type=str, default='CIDER', help='the metric to be monitored.')
  parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

  # Data loader settings
  parser.add_argument('--dataset_name', type=str, default='ffair', help='the dataset to be used.')
  parser.add_argument('--max_seq_length', type=int, default=150, help='the maximum sequence length of the reports.')
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
                      default=0.001, type=float)
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
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
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

  parser.add_argument('--semantic',
                      help='whether use semantic embedding',
                      action='store_true')

  args = parser.parse_args()
  return args

def prepare_class_embedding(split):
  if split == 'train':
    class_for_train = ['1', '3.7', '4.1', '5', '5.1', '6', '7', '9', '10', '10.1', '11', '14', '15', '16', '17', '19', '22', '23', '24', '25', '27', '28', '29', '31', '33', '34', '35', '37', '37.1', '39', '40', '41', '42']
  elif split == 'test':
    class_for_train = ['3', '3.1', '3.4', '3.6', '4.2', '20', '38']
    
  lesion_dict_path = './data/FFAIRNew/lesion_dict.xlsx'
  lesion_dict = pd.read_excel(lesion_dict_path, sheet_name=0)
  all_class_idx = list(str(i).replace(".0", '') for i in lesion_dict['Class'])
  all_class_name = list(str(i) for i in lesion_dict['Name'])
  
  # open class embedding
  class_embedding_path = './data/FFAIRNew/class.json'
  with open(class_embedding_path,'r') as load_f:
    class_embedding = json.load(load_f)
  embeddings = []
  for idx in class_for_train:
    if idx in all_class_idx:
      name_idx = all_class_idx.index(idx)
      name = all_class_name[name_idx]
      embedding = class_embedding[name]
      embeddings.append(embedding)
    else:
      print(idx)
      embeddings.append(np.zeros_like(np.array(embeddings[0]).shape))
  embeddings = np.array(embeddings)
  print(f"prepare {split} class embedding done. (shape {embeddings.shape})")
  return embeddings



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

  output_dir = args.save_dir + "/" + args.net + args.note + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  tokenizer = Tokenizer(args) 
  dataset = FFAIRDataset(args, tokenizer, 'train', return_report=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)

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

  train_class_semantic_embedding = prepare_class_embedding('train')
  test_class_semantic_embedding = prepare_class_embedding('test')
  fasterRCNN = resnet(args.num_class, num_layer, True, args.class_agnostic, None, S_num, U_num)
  fasterRCNN.create_architecture()

  with open("./data/FFAIRNew/vocab.json",'r') as load_f:
    vocab = json.load(load_f)
  entity_names =  []
  with open('./data/FFAIRNew/class.txt', 'r') as f_in:
    for i, input in enumerate(f_in):
      entity_name = input.strip()
      # print(entity_name)
      entity_names.extend([entity_name])
  entity_names = entity_names[:-1]
  print("entity_names", entity_names)

  # create tokenizer
  tokenizer = Tokenizer(args)
  model = FFAIRModel(args, tokenizer).cuda()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr


  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
  optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
  # get function handles of loss and metrics
  criterion = compute_loss
  metrics = compute_scores

  # build optimizer, learning rate scheduler
  rp_optimizer = build_optimizer(args, model)
  lr_scheduler = build_lr_scheduler(args, optimizer)

  if args.cuda:
    fasterRCNN.cuda()

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)
  trainer = Trainer(model, criterion, metrics, args, lr_scheduler, dataloader)
  # iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # trainer.train(epoch, dataloader, fasterRCNN, model, args, test_class_semantic_embedding)
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    step = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    # print("Test")
    trainer.train(epoch, dataloader, fasterRCNN, model, args, test_class_semantic_embedding)
    for data in dataloader:
      loss = 0
      im_data_all, im_info_all, gt_boxes_all, num_boxes_all, targets, targets_masks = data
      im_data_all = im_data_all.squeeze()
      gt_boxes_all = gt_boxes_all[0]
      num_boxes_all = num_boxes_all.squeeze()
      
      # 筛选出有bbox的图像
      index = np.array(num_boxes_all!=0).flatten()
      # print(index)
      im_data_all = im_data_all[index]
      gt_boxes_all = gt_boxes_all[index] # [x, 1, 5]
      num_boxes_all = num_boxes_all[index]

      if args.cuda:
        im_data_batch = im_data_all.cuda() # [b,...]
        im_info_batch = im_info_all.cuda() # [1, 3]
        num_boxes_batch = num_boxes_all.cuda()
        gt_boxes_batch = gt_boxes_all.cuda()
      
      pred_word_idx = []
      fasterRCNN.zero_grad()
      for k in range(len(im_data_batch)):
        im_data = im_data_batch[k].unsqueeze(0)
        im_info = im_info_batch # [1, 3]
        gt_boxes = gt_boxes_batch[k].unsqueeze(0)
        num_boxes = num_boxes_batch[k].unsqueeze(0)
        
        if args.semantic:
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label, att_feats, fc_feats = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, return_feature_and_loss=True, Class_Semantic_embedding=train_class_semantic_embedding)

          # print(cls_prob.shape) # [1, 128, 34]
          if epoch>=10:
            cls_porb_max, cls_prob_max_idx = cls_prob[0].max(dim=1)
            sorted, indices = torch.sort(cls_porb_max,descending=True)
            max_idx = indices[0]
            max_idx = cls_prob_max_idx[max_idx]
            # print(max_idx)
            entity_name = entity_names[max_idx-1]
            # print(entity_name)
            
            entity_name = entity_name.split(' ')
            for word in entity_name:
              if ',' in word:
                word = word.replace(',', '')
              if word in vocab:
                pred_word_idx.append(vocab.index(word)+1) 

        else:
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label, att_feats, fc_feats = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, return_feature_and_loss=True)
          
        
      
        
        loss = loss + rpn_loss_cls.mean() + rpn_loss_box.mean() \
            + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.detach().item()
      
        if k == 0:
            case_att_feats = att_feats.unsqueeze(0) # [1, 16, 2048]
            case_fc_feats = fc_feats.unsqueeze(0) # [1, 2048]
        else:
            case_att_feats = torch.cat([att_feats.unsqueeze(0), case_att_feats],0)
            case_fc_feats = torch.cat([fc_feats.unsqueeze(0), case_fc_feats],0)
        
      # Report Generate ########################################
      reports_ids, reports_masks = targets.squeeze().unsqueeze(0).cuda(), targets_masks.squeeze().unsqueeze(0).cuda()
      output = model(case_att_feats.mean(0).unsqueeze(0),case_fc_feats.mean(0).unsqueeze(0), reports_ids, mode='train')

      rp_loss = criterion(output, reports_ids, reports_masks)
      
      loss = rp_loss + loss

      # backward
      optimizer.zero_grad()
      rp_optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      rp_optimizer.step()

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
    if epoch%5==0:
      trainer.train(epoch, dataloader, fasterRCNN, model, args, test_class_semantic_embedding)
    
    # save_name = os.path.join(output_dir, 'faster_rcnn{}_{}_{}_{}.pth'.format(args.note, args.session, epoch, step))
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'ffair': model.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))
    if epoch%5==0:
      trainer.train(epoch, dataloader, fasterRCNN, model, args)
  if args.use_tfboard:
    logger.close()
