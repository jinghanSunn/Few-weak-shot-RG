import enum
import os
import os.path as osp
import pandas as pd
import numpy as np
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import ipdb
import albumentations

classes = 7
class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None,return_path=False,return_report=False):
        self.root = './data/FFAIRNew/'
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.box_ann_path = os.path.join(self.root, f"{split}_lesion_info_convert.json")
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform_for_non_bbox_image = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.RandomCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])
        self.return_path = return_path
        self.return_report = return_report
        self.height = 224
        self.width = 224
        if split == 'train':
            self.max_box = 8 
        elif split == 'test':
            self.max_box = 20
        elif split == 'val':
            self.max_box = 12
        # self.ann = json.load(open(self.ann_path))
        # self.examples = self.ann[self.split]
        self.ann = pd.read_csv(args.ann_path)
        self.box_ann = json.load(open(self.box_ann_path)) # {"case":{"image":[[xmin,ymin,xmax,ymax,cls]]}}
        self.examples = self.ann
        self.masks = dict()
        self.reports = dict()
        self.image_path = dict()

        for i, each in self.examples.iterrows():
            # self.reports.append(self.tokenizer(each['En_Report'][:self.max_seq_length]))
            self.reports[each['Id']] = self.tokenizer(each['En_Report'][:self.max_seq_length])
            # self.masks.append([1]*len(self.reports[-1]))
            self.masks[each['Id']] = [1]*len(self.reports[each['Id']])
            self.image_path[each['Id']] = each['Image_path']


        box_case_path = os.path.join(self.root, f'{self.split}_case.json') # {"cls":[case],...}
        box_case = json.load(open(box_case_path))
        self.all_bbox_labeled_case = [] 
        for cls in box_case.keys():
            cases = box_case[cls]
            self.all_bbox_labeled_case.extend(cases)

    def __len__(self):
        # return len(self.examples)
        return len(self.all_bbox_labeled_case)
        # return 1

class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        case_id = self.all_bbox_labeled_case[idx]
        image_id = case_id 
        case_bbox_anns = self.box_ann[case_id] 

        image_path = eval(self.image_path[case_id]) 
        images = []
        boxes = []
        num_boxes = [] 
        im_shapes = []
        im_scales = []
        for ind in range(len(image_path)):
            # print("image_path[ind]",image_path[ind])
            image = Image.open(os.path.join(self.image_dir, image_path[ind])).convert('RGB') 
            im_shape = image.size # (512, 512)
            
            im_size_min = np.min(im_shape[0:2])
            im_shapes.append(im_shape)
            im_scales.append(self.height/im_size_min)
            # if self.transform is not None:
            #     images.append(self.transform(image))

            if image_path[ind] in case_bbox_anns.keys():
                image = np.array(image)
                image, box = resize_image_and_box(image, case_bbox_anns[image_path[ind]], self.height, self.width)
                box_len = len(box)
                box = [[x1,y1,x2,y2,c+1] for x1,y1,x2,y2,c in box] # 

                if box_len < self.max_box:
                    box.extend([[0,0,0,0,0]]*(self.max_box-box_len))
                elif box_len > self.max_box: 
                    box = box[:self.max_box]

                images.append(image)
                boxes.append(box)

                num_boxes.extend([len(box)])
            else:
                images.append(self.transform_for_non_bbox_image(image))
                boxes.extend(([[[0,0,0,0,0]]*self.max_box]))
                num_boxes.extend([0])
        
        # print(boxes)
        
        images = torch.stack(images, 0)
        # print(im_scales)
        im_info = [self.height, self.width, im_scales[0]] 

        im_info = torch.Tensor(im_info)
        num_boxes = torch.Tensor(num_boxes)
        boxes = torch.FloatTensor(boxes)


        reports_ids = self.reports[case_id]
        # print(reports_ids)
        reports_masks = self.masks[case_id]
        max_seq_length = len(reports_ids)
        if len(image_path)>150:
            image_id = image_id[:150]
            images = images[:150]
            boxes = boxes[:150]
            num_boxes = num_boxes[:150]

        

        targets = np.zeros((1, max_seq_length), dtype=int)
        targets_masks = np.zeros((1, max_seq_length), dtype=int)


        targets[0, :len(reports_ids)] = reports_ids
        # print(targets)

        # for i, report_masks in enumerate(reports_masks):
        targets_masks[0, :len(reports_masks)] = reports_masks
        
        targets, targets_masks = torch.LongTensor(targets), torch.FloatTensor(targets_masks)


        if self.return_path:
            return images, im_info, boxes, num_boxes, image_path
        if self.return_report:
            return images, im_info, boxes, num_boxes, targets, targets_masks
        return images, im_info, boxes, num_boxes

    def _do_detection_eval(self, results, num_class, U_classes=[]):
        from modules.evaluate import calc_mAP, convert_PRED, convert_GT, filter_det
        pred = convert_PRED(results) 
        GT = convert_GT()
        base_mAP, base_mAR, novel_mAP, novel_mAR = calc_mAP(filter_det(pred, topn=50), GT, num_class, U_classes)
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

    def _results_one_category(self, boxes, all_image_ids):
        results = []
        # print("len(all_image_ids)",len(all_image_ids))
        for im_ind, index in enumerate(all_image_ids):
            dets = np.array(boxes[im_ind]).astype(np.float)
            # print("dets", dets.shape)
            if dets == [] or len(dets.shape)==1: #???why?
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
        # print(len(results))
        return results

    def _get_results(self, all_boxes, all_image_ids, target_classes=[]):
        # classes = self.classes if target_classes==[] else target_classes
        # num_class = len(classes) - 1
        results = {}

        for cls_ind in range(0,classes+1):
        #   if cls == '__background__':
            if cls_ind == 0:
                continue
            # continue
        #   print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes - 1))
            # try:
            print('Collecting {} results'.format(cls_ind))
            results[cls_ind] = self._results_one_category(all_boxes[cls_ind], all_image_ids[cls_ind])
            # except:
            #     print("error", cls_ind)
        return results, classes

    def evaluate_detections(self, all_boxes, output_dir, all_image_ids, target_classes=[], U_classes=[]):
        # print("evaluate_detections all_image_ids", np.array(all_image_ids).shape) # (34, 30000)
        results, num_class = self._get_results(all_boxes, all_image_ids, target_classes)
        json.dump(results, open(osp.join(output_dir, "detections.json"), "w"))
        self._do_detection_eval(results, num_class, U_classes)

def resize_image_and_box(img_arr, bboxes, h, w):
    """
    :param img_arr: original image as a numpy array
    :param bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
    :param h: resized height dimension of image
    :param w: resized weight dimension of image
    :return: dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = albumentations.Compose([
        albumentations.Resize(height=h, width=w, always_apply=True),],
        bbox_params=albumentations.BboxParams(format='pascal_voc')
        )
    totensor = transforms.Compose([         transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),
                                                                (0.229, 0.224, 0.225))])

    transformed = transform(image=img_arr, bboxes=bboxes)

    return totensor(transformed['image']), [list(i) for i in transformed['bboxes']]


    



