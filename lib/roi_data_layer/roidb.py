"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
import os
import pickle

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb
import ipdb

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  # print("roidb",roidb)
  # if not (imdb.name.startswith('coco')):
  #   cache_file = os.path.join(imdb.cache_path, imdb.name + '_sizes.pkl')
  #   if os.path.exists(cache_file):
  #     print('Image sizes loaded from %s' % cache_file)
  #     with open(cache_file, 'rb') as f:
  #       sizes = pickle.load(f)
  #   else:
  #     print('Extracting image sizes... (It may take long time)')
  #     sizes = [PIL.Image.open(imdb.image_path_at(i)).size
  #               for i in range(imdb.num_images)]
  #     with open(cache_file, 'wb') as f:
  #       pickle.dump(sizes, f)
  #     print('Done!!')
         
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    # if not (imdb.name.startswith('coco')):
    #   roidb[i]['width'] = sizes[i][0]
    #   roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps # overlap好像都是1
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True, novel_classes=[]):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if training and cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb):
    print('Loaded dataset `{:s}`'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # gt?
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  imdb_names = imdb_names.split('+') # imdb文件时convert_coco后得到的，有width，height，path，ann
  print("imdb_names",imdb_names)
  if len(imdb_names)==1:
    imdb_name = imdb_names[0]
    imdb = get_imdb(imdb_name, novel_classes) # 获取数据集iDataset
    ipdb.set_trace()
    roidb = get_roidb(imdb)
  else:
    roidbs = [get_roidb(s) for s in imdb_names]
    roidb = roidbs[0]
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1], novel_classes)
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)

  if training:
    roidb = filter_roidb(roidb)
  # print("roidb", roidb)
  '''
  {
  'img_id'
  'img'
  'width': width,
  'height': height,
  'boxes': boxes,
  'gt_classes': gt_classes,
  'gt_overlaps': overlaps,
  'flipped': False,
  'seg_areas': seg_areas}
  '''
  ratio_list, ratio_index = rank_roidb_ratio(roidb) # box高度和宽度的比例

  return imdb, roidb, ratio_list, ratio_index
