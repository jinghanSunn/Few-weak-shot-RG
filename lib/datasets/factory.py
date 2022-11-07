from datasets.mydataset import iDataset

import numpy as np

__sets = {
  "coco6515_2014_train": (lambda novel_classes=[]: iDataset("coco6515", "train", "2014", novel_classes=novel_classes)),
  "coco6515_2014_test": (lambda novel_classes=[]: iDataset("coco6515", "test", "2014", novel_classes=novel_classes)),
  "voc164_0712_train": (lambda novel_classes=[]: iDataset("voc164", "train", "0712", novel_classes=novel_classes)),
  "voc164_2007_test": (lambda novel_classes=[]: iDataset("voc164", "test", "2007", novel_classes=novel_classes)),
  "voc164_2007_testdet": (lambda novel_classes=[]: iDataset("voc164", "testdet", "2007", novel_classes=novel_classes)),
  "vg478130_2017_train": (lambda novel_classes=[]: iDataset("vg478130", "train", "2017", novel_classes=novel_classes)),
  "vg478130_2017_test": (lambda novel_classes=[]: iDataset("vg478130", "test", "2017", novel_classes=novel_classes))
}

def get_imdb(name, novel_classes=[]):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  if novel_classes==[]:
    return __sets[name]()
  else:
    return __sets[name](novel_classes=novel_classes)


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
