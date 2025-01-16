#region: Chưa xong nha

from collections import Counter
from Levenshtein import distance
from yolov5.utils.augmentations import letterbox

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_LP_OCR = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')

for param in yolo_LP_detect.model.model.parameters():
    param.requires_grad = False

for param in yolo_LP_OCR.model.model.parameters():
    param.requires_grad = False

# Intersection over Union score for Object Detection
def IoU(truth_bbox, pred_bbox):
    # input order is xmin, xmax, ymin, ymax
    # x2 = xmax, x1 = xmin
    x1_true, x2_true, y1_true, y2_true = truth_bbox
    x1_pred, x2_pred, y1_pred, y2_pred = pred_bbox

    xA = max(x1_true, x1_pred)
    yA = max(y1_true, y1_pred)

    xB = min(x2_true, x2_pred)
    yB = min(y2_true, y2_pred)

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    trueArea = (x2_true - x1_true) * (y2_true - y1_true)
    predArea = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    unionArea = trueArea + predArea - interArea

    iou = interArea / unionArea
    return iou

def similarity_metric(str1, str2):
    # Normalize the strings by removing non-alphanumeric characters and converting to uppercase
    str1 = ''.join(filter(str.isalnum, str1)).upper()
    str2 = ''.join(filter(str.isalnum, str2)).upper()

    # Count the occurrences of each character in both strings
    count_str1 = Counter(str1)
    count_str2 = Counter(str2)

    # Calculate the match count based on the minimum number of appearances of each character
    match_count = sum(min(count_str1[char], count_str2[char]) for char in count_str2)

    # Calculate the similarity based on the length of the second string
    similarity = match_count / len(str1)

    return similarity

def cer_metric(ground_truth, pred):
    edit_operations = distance(ground_truth, pred)
    cer = edit_operations / len(ground_truth)
    return cer

yolo_LP_OCR.conf = 0.60

def get_bbox(labels, multiboxes = False):
    with open(labels, 'r') as file:
      if multiboxes:
        bboxes = []
        for line in file:
          line = line.strip().split(' ')[1:]
          bboxes.append([float(i) for i in line])
        return bboxes
      else:
        first_line = file.readline().strip().split(' ')[1:]
        bbox = [float(i) for i in first_line]
        return bbox

def prepare_bbox_afer_attack(img_path, bbox, new_img_size=640):
    origin_img_path = img_path.replace('attack_results', 'LP_detection')
    origin_img = cv2.imread(origin_img_path)
    # kích thước gốc hình chưa pad
    or_h, or_w, _ = origin_img.shape
    # tọa độ gốc ở ảnh chưa padding
    xc, yc, w, h = bbox
    # lấy ra tỉ lệ giãn nỡ của hình gốc và hình sau khi padding
    pad_img, ratio, pad = letterbox(origin_img, new_img_size, auto=False, scaleup=True)
    # chuyển từ tọa độ gốc sang tọa độ sau khi padding
    xc = xc * or_w * ratio[0] + pad[0]
    yc = yc * or_h * ratio[1] + pad[1]
    w = w * or_w * ratio[0]
    h = h * or_h * ratio[1]
    xmin = (xc - w / 2)
    xmax = (xc + w / 2)
    ymin = (yc - h / 2)
    ymax = (yc + h / 2)
    return xmin, xmax, ymin, ymax

def prepare_bbox_for_blurring(bbox, w, h):
  # return xmin, xmax, ymin, ymax
  xmin = (int)((bbox[0] - bbox[2] / 2) * w)
  xmax = (int)((bbox[0] + bbox[2] / 2) * w)
  ymin = (int)((bbox[1] - bbox[3] / 2) * h)
  ymax = (int)((bbox[1] + bbox[3] / 2) * h)

  return xmin, xmax, ymin, ymax

#@title

#enter image path here
def evaluate_detect_single_img(img_path, multible_boxes = False, verbose=False, attacked=False):
  label_path = img_path.replace(".jpg", ".txt")
  label_path = label_path.replace("images", "labels")
  cv2_img = cv2.imread(img_path)
  h, w, _ = cv2_img.shape
  bbox = get_bbox(label_path)
  yolo_LP_detect.eval()
  if attacked:
    truth_scaled_bbox = prepare_bbox_afer_attack(img_path, bbox)
  else:
    truth_scaled_bbox = prepare_bbox_for_blurring(bbox, w, h)
  plates = yolo_LP_detect(cv2_img, size=640)
  list_plates = plates.pandas().xyxy[0].values.tolist()
  if multible_boxes:
    # trường hợp detect được nhiều biển số trong ảnh
    return None
  else:
    # chỉ kiểm tra 1 biển số xe trong hình ảnh
    if verbose:
        # visualize_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bbox, w, h)
        img = io.imread(img_path) #Read the image
        fig = px.imshow(img)
    if len(list_plates):
      plate = list_plates[0]
      if verbose:
        # visualize_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bbox, w, h)
        fig.add_shape(type='rect',x0=truth_scaled_bbox[0], x1=truth_scaled_bbox[1], y0=truth_scaled_bbox[2], y1=truth_scaled_bbox[3], xref='x', yref='y',line_color='red')
      fig.show()
      pred_bbox = [plate[0], plate[2], plate[1], plate[3]]
      print("truth: ", truth_scaled_bbox)
      print("pred bbox: ", pred_bbox)
      iou = IoU(truth_scaled_bbox, pred_bbox)
      return iou
    else:
      fig.show()
      print("cannot predict")
      return 0
    
#region Detect Yolov5

imgs_folder = f'LP_detection_{12}/images/train'
paths = f"/content/{imgs_folder}"
img_files = sorted(os.listdir(paths))
origin_image_path = os.path.join(paths, img_files[0])

origin_pred_iou = evaluate_detect_single_img(origin_image_path, verbose=True)

print(f"Origin image IoU: {origin_pred_iou}")

#region Detect unauthorized

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)


# ---------- model
inception_resnet_model = Model(inputs=inception_resnet.input,outputs=headmodel)
inception_resnet_model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

