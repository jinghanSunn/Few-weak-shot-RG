from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
# import scipy.io as sio
import pickle
import json

# {
# "images": {
#   image_id: {width, height, path, anns:[[class_id, [x, y, w, h]], ...]}
# }, 
# "classes":[{class_id, classname}, ...]
# }

class iDataset(imdb):
  def __init__(self, data_name, image_set, year, novel_classes=[]):
    # generate ROIs without novel classes
    imdb.__init__(self, data_name + '_' + year + '_' + image_set)
    # name, paths
    self._year = year
    self._image_set = image_set
    self._data_path = cfg.DATA_DIR

    self._novel_classes = novel_classes
    self._dataset = json.load(open(osp.join(self._data_path, "annotations/%s_%s_%s.json"%(data_name, year, image_set))))
    cats = self._dataset['classes']
    base_cats = list(filter(lambda c:c['name'] not in self._novel_classes, cats))
    self._classes = tuple(['__background__'] + [c['name'] for c in base_cats])
    self._class_to_ind = dict(zip(self.classes, list(range(self.num_classes))))
    self._class_to_clsid = dict(zip([c['name'] for c in cats], [int(c['class_id']) for c in cats]))
    self._base_ids = list(map(lambda x:self._class_to_clsid[x['name']], base_cats))
    self._novel_ids = list(map(lambda x:self._class_to_clsid[x], self._novel_classes))
    image_index = list(self._dataset['images'].keys())
    self._image_index = sorted(image_index)
    
    # Default to roidb handler
    self.set_proposal_method('gt')
    self.competition_mode(False)

    # Dataset splits that have ground-truth annotations
    self._gt_splits = ('train', 'val', 'trainval')

  
  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = self._dataset['images'][index]['path']
    image_path = osp.join(self._data_path, file_name)
    assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
    return image_path

  def _is_valid(self, index, withnovel=True):
    objs = self._dataset['images'][index]['anns']
    clsids = list(map(lambda x:x[0], objs))
    clsset = set(clsids)
    if withnovel:
      # Check whether an image contains base objects
      if not clsset.isdisjoint(set(self._base_ids)):
          return True
    else:
      # Check whether an image contains base objects only
      if clsset.isdisjoint(set(self._novel_ids)):
        return True 
    return False

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """        
    if "test" in self._image_set:
      gt_roidb = [self._load_annotation(index) for index in self._image_index]
      return gt_roidb

    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_annotation(index) for index in self._image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_annotation(self, index):
    im_ann = self._dataset['images'][index]
    width = im_ann['width']
    height = im_ann['height']
    objs = im_ann['anns']
    
    valid_objs = []
    for obj in objs:
      if int(obj[0]) in self._novel_ids:
        continue
      bbox = obj[1]
      obj.append(bbox[2]*bbox[3])
      x1 = np.max((0, bbox[0]))
      y1 = np.max((0, bbox[1]))
      x2 = np.min((width - 1, x1 + np.max((0, bbox[2] - 1))))
      y2 = np.min((height - 1, y1 + np.max((0, bbox[3] - 1)))) #x1 y1：左上坐标；x2,y2:右下坐标
      if bbox[2]*bbox[3] > 0 and x2 >= x1 and y2 >= y1:
        obj.append([x1, y1, x2, y2])
        valid_objs.append(obj)
    objs = valid_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Lookup table to map from COCO category ids to our internal class indices
    clsid_to_class_ind = dict([(self._class_to_clsid[cls], self._class_to_ind[cls]) for cls in self._classes[1:]])

    for ix, obj in enumerate(objs):
      cls = clsid_to_class_ind[int(obj[0])]
      boxes[ix, :] = obj[3]
      gt_classes[ix] = cls
      seg_areas[ix] = obj[2]
      #if obj['iscrowd']: overlaps[ix, :] = -1.0
      overlaps[ix, cls] = 1.0

    ds_utils.validate_boxes(boxes, width=width, height=height)
    overlaps = scipy.sparse.csr_matrix(overlaps)
    return {'width': width,
            'height': height,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_widths(self):
    return [r['width'] for r in self.roidb]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


  def _do_detection_eval(self, results, num_class, U_classes=[]):
    from datasets.evaluate import calc_mAP, convert_PRED, convert_GT, filter_det
    pred = convert_PRED(results)
    GT = convert_GT(self._dataset)
    base_mAP, base_mAR, novel_mAP, novel_mAR = calc_mAP(filter_det(pred, topn=100), GT, num_class, U_classes)
    if U_classes==[]:
      print("mAP of BaseCLS: %.2f, Recall@100: %.2f"%(base_mAP, base_mAR))
    elif base_mAP==0:
      print("mAP of NovelCLS: %.2f, Recall@100: %.2f"%(novel_mAP, novel_mAR))
    else:
      hm = 2*base_mAP*novel_mAP/(base_mAP+novel_mAP)
      hmrec = 2*base_mAR*novel_mAR/(base_mAR+novel_mAR)
      print("mAP: BaseCLS-%.2f, NovelCLS-%.2f, HM-%.2f, \
        Recall@100: BaseCLS-%.2f, NovelCLS-%.2f, HM-%.2f"%(base_mAP, novel_mAP, hm, 
          base_mAR, novel_mAR, hmrec))

  def _results_one_category(self, boxes):
    results = []
    for im_ind, index in enumerate(self.image_index):
      dets = boxes[im_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] #- xs + 1
      hs = dets[:, 3] #- ys + 1
      results.extend(
        [{'image_id': index,
          'bbox': [xs[k], ys[k], ws[k], hs[k]],
          'C': scores[k]} for k in range(dets.shape[0])])
    return results

  def _get_results(self, all_boxes, target_classes=[]):
    classes = self.classes if target_classes==[] else target_classes
    num_class = len(classes) - 1
    results = {}
    for cls_ind, cls in enumerate(classes):
      if cls == '__background__':
        continue
      print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes - 1))
      results[cls] = self._results_one_category(all_boxes[cls_ind])

    return results, num_class

  def evaluate_detections(self, all_boxes, output_dir, target_classes=[], U_classes=[]):
    results, num_class = self._get_results(all_boxes, target_classes)
    json.dump(results, open(osp.join(output_dir, "detections.json"), "w"))
    self._do_detection_eval(results, num_class, U_classes)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)


