import json
from collections import defaultdict
from tqdm import tqdm


post = "_65_15" # "_48_17" # 
with open('data/categories_coco%s.json'%post) as f:
    classes = json.load(f)
    all_classes = classes['seen']+classes['unseen']


base_ids = list(range(len(classes['seen'])))
novel_ids = list(range(len(classes['seen']), len(all_classes)))


def is_valid(objs, withnovel=False):
    clsids = list(map(lambda x:x[0], objs))
    clsset = set(clsids)
    # Check whether an image contains base objects only
    if clsset.isdisjoint(set(novel_ids)):
        return True 
    return False

img2boxs = defaultdict(list)
img_info = {}
with open('/apdcephfs/private_jinghansun/detectron2-main/tools/datasets/coco/annotations/instances_train2017.json', encoding='utf-8') as f:
    data = json.load(f)
    annotations = data['annotations']
    category = data['categories']
    print("Before filtering: ", len(data['images']))

    id2name = {}
    for c in category:
        id2name[c['id']] = c['name']

    for item in data['images']:
        img_info[item['id']] = {'width':item['width'], 'height':item['height']}

    for ant in tqdm(annotations):
        img = ant['image_id']
        name = img
        cat = ant['category_id']
        cat_name = id2name[cat]
        if cat_name in all_classes:
            img2boxs[name].append([all_classes.index(cat_name), ant['bbox']])

img2boxs = {k: v for k, v in img2boxs.items() if is_valid(v)}
img_info = {k: v for k, v in img_info.items() if k in img2boxs}
print("After filtering: ", len(img2boxs))

out_json = { "images": {}, "classes": [] }
for cls_idx, name in enumerate(all_classes):
    out_json["classes"].append({"class_id": str(cls_idx), "name": name})
for image_id in img2boxs.keys():
    out_json["images"][image_id] = {}
    filepath = '/apdcephfs/private_jinghansun/detectron2-main/tools/datasets/coco/train2017/%012d.jpg'%image_id
    anns = img2boxs[image_id]
    out_json["images"][image_id]["width"] = img_info[image_id]["width"]
    out_json["images"][image_id]["height"] = img_info[image_id]["height"]
    out_json["images"][image_id]["path"] = filepath
    out_json["images"][image_id]["anns"] = anns
json.dump(out_json, open("coco6515_2014_train.json", "w"))
'''
 "185925": {"width": 500, "height": 290, "path": "/apdcephfs/private_jinghansu
n/detectron2-main/tools/datasets/coco/train2017/000000185925.jpg", "anns": [[24, [352.93, 
33.78, 12.26, 12.84]], [0, [254.16, 80.16, 221.57, 203.97]], [30, [297.79, 168.09, 47.36, 
44.77]]]}
'''


def is_valid(objs):
    clsids = list(map(lambda x:x[0], objs))
    clsset = set(clsids)
    # Check whether an image contains novel objects
    if not clsset.isdisjoint(set(novel_ids)):
        return True
    return False

img2boxs = defaultdict(list)
img_info = {}
with open('/apdcephfs/private_jinghansun/detectron2-main/tools/datasets/coco/annotations/instances_val2017.json', encoding='utf-8') as f:
    data = json.load(f)
    annotations = data['annotations']
    category = data['categories']
    print("Before filtering: ", len(data['images']))

    id2name = {}
    for c in category:
        id2name[c['id']] = c['name']

    for item in data['images']:
        img_info[item['id']] = {'width':item['width'], 'height':item['height']}

    for ant in tqdm(annotations):
        img = ant['image_id']
        name = img
        cat = ant['category_id']
        cat_name = id2name[cat]
        if cat_name in all_classes:
            img2boxs[name].append([all_classes.index(cat_name), ant['bbox']])

img2boxs = {k: v for k, v in img2boxs.items() if is_valid(v)}
img_info = {k: v for k, v in img_info.items() if k in img2boxs}
print("After filtering: ", len(img2boxs))

cnt = 0
out_json = { "images": {}, "classes": [] }
for cls_idx, name in enumerate(all_classes):
    out_json["classes"].append({"class_id": str(cls_idx), "name": name})
for image_id in img2boxs.keys():
    out_json["images"][image_id] = {}
    filepath = '/apdcephfs/private_jinghansun/detectron2-main/tools/datasets/coco/val2017/%012d.jpg'%image_id
    anns = img2boxs[image_id]
    out_json["images"][image_id]["width"] = img_info[image_id]["width"]
    out_json["images"][image_id]["height"] = img_info[image_id]["height"]
    out_json["images"][image_id]["path"] = filepath
    out_json["images"][image_id]["anns"] = anns
    cnt += len(anns)
json.dump(out_json, open("coco6515_2014_test.json", "w"))


# res_cocoformat = {'images':[], 'annotations':[]}
# res_cocoformat['categories'] = data['categories']
# for item in data['images']:
#     if item['id'] in img_info:
#         res_cocoformat['images'].append(item)
# for item in data['annotations']:
#     if item['image_id'] in img_info:
#         res_cocoformat['annotations'].append(item)
# print(len(res_cocoformat['images']), len(res_cocoformat['annotations']), cnt)
# json.dump(res_cocoformat, open("coco6515_2014_test_cocoformat.json", "w"))

'''
def is_valid(objs):
    clsids = list(map(lambda x:x[0], objs))
    clsset = set(clsids)
    # Check whether an image contains base objects
    if not clsset.isdisjoint(set(base_ids)):
        return True
    return False
img2boxs = defaultdict(list)
img_info = {}
with open('data/coco/annotations/instances_val2014.json', encoding='utf-8') as f:
    data = json.load(f)
    annotations = data['annotations']
    category = data['categories']
    print("Before filtering: ", len(data['images']))

    id2name = {}
    for c in category:
        id2name[c['id']] = c['name']

    for item in data['images']:
        img_info[item['id']] = {'width':item['width'], 'height':item['height']}

    for ant in tqdm(annotations):
        img = ant['image_id']
        name = img
        cat = ant['category_id']
        cat_name = id2name[cat]
        if cat_name in all_classes:
            img2boxs[name].append([all_classes.index(cat_name), ant['bbox']])

img2boxs = {k: v for k, v in img2boxs.items() if is_valid(v)}
img_info = {k: v for k, v in img_info.items() if k in img2boxs}
print("After filtering: ", len(img2boxs))

out_json = { "images": {}, "classes": [] }
for cls_idx, name in enumerate(all_classes):
    out_json["classes"].append({"class_id": str(cls_idx), "name": name})
for image_id in img2boxs.keys():
    out_json["images"][image_id] = {}
    filepath = 'coco/images/val2014/COCO_val2014_%012d.jpg'%image_id
    anns = img2boxs[image_id]
    out_json["images"][image_id]["width"] = img_info[image_id]["width"]
    out_json["images"][image_id]["height"] = img_info[image_id]["height"]
    out_json["images"][image_id]["path"] = filepath
    out_json["images"][image_id]["anns"] = anns
json.dump(out_json, open("coco6515_2014_testdet.json", "w"))
'''