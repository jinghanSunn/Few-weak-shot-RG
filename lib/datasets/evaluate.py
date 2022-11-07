# coding=utf-8
import json
import sys
import os
import numpy as np
import time


# cats = {
#     "seen":["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
#     "unseen":["cymbal", "piano", "trumpet", "lantern", "life saver", "hockey", "fish", "camel", "pigeon", "tomato", "pepper", "pumpkin", "lemon", "strawberry", "pear", "bread", "canned", "meat balls", "carriage", "donkey", "printer", "converter", "microphone", "stapler", "pen/pencil", "scale", "stool", "gas stove", "vent", "washing machine", "barrel/bucket", "toothpaste", "cabinet/shelf", "air conditioner", "plate", "clutch", "hat", "luggage", "necklace", "trophy"]
# }
# cls2cid = {"scissors": "75", "tennis racket": "37", "carriage": "118", "stop sign": "11", "fire hydrant": "10", "giraffe": "23", "horse": "17", "trumpet": "102", "keyboard": "65", "traffic light": "9", "car": "2", "surfboard": "36", "lemon": "112", "umbrella": "25", "barrel/bucket": "130", "chair": "55", "microphone": "122", "cake": "54", "pumpkin": "111", "pizza": "52", "orange": "48", "cow": "19", "piano": "101", "converter": "121", "parking meter": "12", "dining table": "59", "backpack": "24", "tomato": "109", "tie": "27", "zebra": "22", "bed": "58", "pepper": "110", "donut": "53", "vase": "74", "scale": "125", "meat balls": "117", "stool": "126", "sandwich": "47", "suitcase": "28", "cell phone": "66", "oven": "68", "pen/pencil": "124", "bottle": "38", "motorcycle": "3", "snowboard": "31", "wine glass": "39", "sink": "70", "carrot": "50", "stapler": "123", "bench": "13", "boat": "8", "sports ball": "79", "knife": "42", "baseball glove": "34", "remote": "64", "hot dog": "51", "toothbrush": "78", "mouse": "63", "refrigerator": "71", "necklace": "138", "plate": "134", "train": "6", "clutch": "135", "bowl": "44", "bread": "115", "microwave": "67", "toothpaste": "131", "clock": "73", "printer": "120", "elephant": "20", "skis": "30", "gas stove": "127", "life saver": "104", "broccoli": "49", "lantern": "103", "bird": "14", "pear": "114", "toilet": "60", "luggage": "137", "vent": "128", "handbag": "26", "trophy": "139", "laptop": "62", "cup": "40", "person": "0", "pigeon": "108", "bicycle": "1", "hockey": "105", "canned": "116", "sheep": "18", "dog": "16", "tv": "61", "cat": "15", "toaster": "69", "spoon": "43", "airplane": "4", "kite": "32", "couch": "56", "fork": "41", "potted plant": "57", "frisbee": "29", "teddy bear": "76", "cabinet/shelf": "132", "washing machine": "129", "cymbal": "100", "fish": "106", "baseball bat": "33", "strawberry": "113", "hair drier": "77", "skateboard": "35", "bear": "21", "banana": "45", "bus": "5", "hat": "136", "camel": "107", "donkey": "119", "truck": "7", "book": "72", "apple": "46", "air conditioner": "133"}



def calc_ap(rec, prec):
    if False:
        # VOC 07 11 point method
        rec = np.array(rec)
        prec = np.array(prec)
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    mrec = np.array([0, *rec, 1])
    mpre = np.array([0, *prec, 0])

    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    idx1 = np.where(mrec[:-1] != mrec[1:])[0]
    idx2 = idx1 + 1

    ap = ((mrec.take(idx2) - mrec.take(idx1)) * mpre.take(idx2)).sum()
    return ap

def convert_GT(anns):
    """
    {
        "images": {
            image_id: {width, height, path, anns:[[class_id, [x, y, w, h]], ...]}
        }, 
        "classes":[{class_id, classname}, ...]
    }
    {
        obj_per_image: {file_id: [{cls: "classname", bbox: [xmin, ymin, xmax, ymax]}, ...], ...},
        counter_per_class: {class_name: 100, ...},
        counter_per_image: {file_id: 20, ...}
    }
    """
    cid2cls = dict(zip([int(c['class_id']) for c in anns['classes']], [c['name'] for c in anns['classes']]))
    res = {
        'obj_per_image': {},
        'counter_per_class': {},
        'counter_per_image': {}
    }
    for image_id in anns['images']:
        anns_img = anns['images'][image_id]
        res['counter_per_image'][image_id] = len(anns_img['anns'])
        gts_img = []
        for ann in anns_img['anns']:
            class_id = int(ann[0])
            cls_name = cid2cls[class_id]
            bbox = ann[1]
            bbox = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
            gts_img.append({'cls':cls_name, 'bbox':bbox})
            if cls_name not in res['counter_per_class']:
                res['counter_per_class'][cls_name] = 0
            res['counter_per_class'][cls_name] += 1
        res['obj_per_image'][image_id] = gts_img
    return res


def convert_PRED(pred):
    """
    {
        class_name: [{image_id: "123.jpg", bbox: [xmin, ymin, xmax, ymax], C: 1.0}, ...], ...
    }
    """
    for cls_name in pred:
        pred[cls_name].sort(key=lambda x:x["C"], reverse=True)
    return pred


MINOVERLAP = 0.5
def calc_mAP(submit, GT, num_class, U_classes=[]):
    """
    {
        class_name: [{image_id: "123.jpg", bbox: [xmin, ymin, xmax, ymax], C: 1.0}, ...], ... # 置信度降序
    }
    {
        obj_per_image: {file_id: [{cls: "cls0", bbox: [xmin, ymin, xmax, ymax]}, ...], ...},
        counter_per_class: {class_name: 100, ...},
        counter_per_image: {file_id: 20, ...}
    }
    """
    # print(submit['0'][0])
    # print(GT['obj_per_image']['000001'])
    # print(xx.xx)
    gt_counter_per_class = GT["counter_per_class"]
    gt_counter_per_image = GT["counter_per_image"]

    used = {}
    for file in gt_counter_per_image:
        used[file] = [False]*gt_counter_per_image[file]
    base_sum_AP = 0.
    novel_sum_AP = 0.
    base_sum_AR = 0.
    novel_sum_AR = 0.
    count_true_positives = {}
    base_cls_num = 0

    for class_name in gt_counter_per_class:
        count_true_positives[class_name] = 0
        if class_name not in submit:
            continue
        dr_data = submit[class_name]
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            file_id = detection["image_id"]
            if file_id not in GT["obj_per_image"]:
                # fp[idx] = 1
                continue
            # assign detection-results to ground truth object if any
            ground_truth_data = GT["obj_per_image"][file_id]
            ovmax = -1
            gt_match = -1
            gt_match_idx = -1
            # load detected object bounding-box
            bb = detection["bbox"]
            for idx_obj, obj in enumerate(ground_truth_data):
                # look for a class_name match
                if obj["cls"] == class_name:
                    bbgt = obj["bbox"]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    iw = max(iw, 0.)
                    ih = max(ih, 0.)
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj
                            gt_match_idx = idx_obj
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if not used[file_id][gt_match_idx]:
                    # true positive
                    tp[idx] = 1
                    used[file_id][gt_match_idx] = True
                    count_true_positives[class_name] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:] # deep copy
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        
        ar = sum(rec)/len(rec)
        # the same Average Recall calclulation as 
        # Shafin Rahman, Salman Khan, and Nick Barnes. "Polarity Loss for Zero-shot Object Detection."
        # and Shafin Rahman, Salman Khan, and Nick Barnes. "Improved Visual-Semantic Alignment for Zero-Shot Object Detection," AAAI 2020.
        # https://github.com/salman-h-khan/PL-ZSD_Release/blob/43e503ebb56ad0afb83345573e0843bdf63991da/keras_retinanet/utils/eval.py#L285
        ap = calc_ap(rec[:], prec[:])
        if class_name in U_classes:
            novel_sum_AP += ap
            novel_sum_AR += ar
        else:
            base_sum_AP += ap
            base_sum_AR += ar
            base_cls_num += 1

        print("%s%s: AP-%.2f, REC-%.2f"%("NovelCLS " if class_name in U_classes else "", class_name, ap*100, ar*100))

    cls_num = num_class
    novel_cls_num = len(U_classes)
    base_cls_num = cls_num - novel_cls_num
    base_mAP = 0 if base_cls_num==0 else base_sum_AP / base_cls_num
    novel_mAP = 0 if novel_cls_num==0 else novel_sum_AP / novel_cls_num
    base_mAR = 0 if base_cls_num==0 else base_sum_AR / base_cls_num
    novel_mAR = 0 if novel_cls_num==0 else novel_sum_AR / novel_cls_num
    
    return base_mAP*100, base_mAR*100, novel_mAP*100, novel_mAR*100

def filter_det(data, confidence=0.000001, topn=100):
    temp = {}
    for clsname, dets in data.items():
        for det in dets:
            confi = det['C']
            img = det['image_id']
            if confi<confidence:
                continue
            if img not in temp:
                temp[img] = []
            temp[img].append({
                "cls":clsname, 
                "bbox":det['bbox'], 
                "C":confi
            })
    for key in temp:
        temp[key].sort(key=lambda x:x["C"], reverse=True)
    for key, value in temp.items():
        if len(value)>topn:
            temp[key] = value[:topn]

    result = {}
    for key, value in temp.items():
        for item in value:
            cls = item['cls']
            if cls not in result:
                result[cls] = []
            result[cls].append({
                "image_id":key, 
                "bbox":item['bbox'], 
                "C":item['C']
            })
    for key in result:
        result[key].sort(key=lambda x:x["C"], reverse=True)
    return result

def convert_det(data, topn=100):
    all_dets = []
    for k,v in data.items():
        all_dets += v
    img_dets = {}
    for det in all_dets:
        img = det['image_id']
        if img not in img_dets:
            img_dets[img] = []
        img_dets[img].append(det)
    for img in img_dets:
        img_dets[img].sort(key=lambda x:x["C"], reverse=True)
    for key, value in img_dets.items():
        if len(value)>topn:
            img_dets[key] = value[:topn]

    res = []
    for k,v in img_dets.items():
        res += v
    res.sort(key=lambda x:x["C"], reverse=True)
    return res

if __name__=="__main__":
    
    pred_path = sys.argv[1]
    anns_path = sys.argv[2]

    with open(pred_path, 'r') as pred_f:
        pred = json.load(pred_f)

    with open(anns_path, 'r') as anns_f:
        anns = json.load(anns_f)

    try:
        pred = convert_PRED(pred)
        GT = convert_GT(anns)
        score = calc_mAP(pred, GT)
        print(score)
    except:
        print("Error")
