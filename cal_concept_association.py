import json
import numpy as np
import copy

def cosine_similarity(x1, x2, eps=1e-8):
  # x1: #cls1 x emb_dim
  # x2: emb_dim x #cls2
  dot12 = np.matmul(x1, x2) #-> #cls1 x #cls2
  w1 = np.linalg.norm(x1, ord=2, axis=1, keepdims=True) #-> #cls1 x 1
  w2 = np.linalg.norm(x2, ord=2, axis=0, keepdims=True) #-> 1 x #cls2
  w12 = np.matmul(w1, w2).clip(min=eps, max=1e8) #-> #cls1 x #cls2
  return (dot12/w12)


with open('data/categories_coco_65_15.json') as f:
    categories = json.load(f)
embeddings_json = json.load(open('data/embeddings_w2v.json'))


all_cats = categories['seen'] + categories['unseen']
embeddings = np.zeros([len(all_cats), len(embeddings_json['car'])])
for idx, cat in enumerate(all_cats):
    embeddings[idx] = embeddings_json[cat]

centroids = copy.deepcopy(embeddings[len(categories['seen']):])
seenembs = embeddings[:len(categories['seen'])]

K = 5
THRESH = 0.1
def targetCentricAssociation(cls2asso):
    for idx, cat in enumerate(categories['seen']):
        emb = embeddings[idx].reshape((-1, 1))
        sims = cosine_similarity(centroids, emb).reshape((-1))
        # multiple association
        topk = np.argsort(sims, -1)[::-1]
        sims[sims<0.1] = 0
        sims[topk[K:]] = 0
        cls2asso[cat] = [float(item) for item in sims]
    return cls2asso


cls2asso = targetCentricAssociation({})
for idx, cat in enumerate(categories['unseen']):
    cls2asso[cat] = int(idx)

post = "_dist_%d"%K
json.dump(cls2asso, open("cls2asso_coco_w2v%s.json"%post, "w"))
# move the result to data/cls2asso_xxx.json