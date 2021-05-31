import numpy as np

def compare_mae(img_true, img_test):
  img_true = img_true.astype(np.float32)
  img_test = img_test.astype(np.float32)
  return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)