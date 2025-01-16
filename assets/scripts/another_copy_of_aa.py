# %% [markdown]
# # Adversarial Attack on License Plate Detecting and OCR

# %%
!nvidia-smi

# %% [markdown]
# ## Prepare Data

# %%
!gdown --help
!unzip --help
!rm --help
!pip --help

# %%
# train - original
!gdown -q --folder https://drive.google.com/drive/folders/1D7ARqpfnFUNbqIkPX57eQXJyJdLeLzz-?usp=drive_link -O /content/

# %%
# train - attacked
!gdown -q --folder https://drive.google.com/drive/folders/1Y4PBzWN88xlUCE8xudYwldv751UI-DmQ?usp=drive_link -O /content/

# %%
# val
!gdown -q --folder https://drive.google.com/drive/folders/1H-KTqNlk2qDAOoFkAmzEhpno8nq-SrjD?usp=drive_link -O /content/

# %%
%cd /content
!unzip -q /content/train_zip/LP_detection_12.zip
!unzip -q /content/train/attacked_imges_train_split_12.zip -d /content/attacked_images/

# %%
%cd /content
!unzip -q /content/train_zip/LP_detection_16.zip
!unzip -q /content/train/attacked_imges_train_split_16.zip -d /content/attacked_images/attacked_imges_train_split_16/

# %%
%cd /content
!gdown --fuzzy https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view
!unzip -q LP_detection.zip
!rm LP_detection.zip

# %%
import shutil
shutil.rmtree(f'/content/LP_detection_{15}')
shutil.rmtree('/content/attacked_images')

# %%
# Download dataset using gdown (cli)
%cd /content
!gdown --fuzzy https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view
!unzip -q LP_detection.zip
!rm LP_detection.zip

# %%
# !gdown --fuzzy https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view
# !unzip -q OCR.zip
# !rm OCR.zip

# %%
# Use utils code from our repo
!git clone https://github.com/steveNguyen1206/mhud.git

# %%
%cd /content/mhud
from matplotlib import pyplot as plt
import plotly.express as px
from tqdm.auto import tqdm
from skimage import io
import pandas as pd
import shutil
import cv2
import os
from attack.deid import Blur, Pixelate

# %% [markdown]
# ### Testing de-identification on image

# %%
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

def visualize_bbox(image, bbox, w, h):
  fig = px.imshow(image)
  fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Original image with bounding box')
  fig.add_shape(type='rect',x0=(bbox[0] - bbox[2] / 2)* w, x1=(bbox[0] + bbox[2] / 2)* w, y0=(bbox[1] - bbox[3]/2) * h, y1=(bbox[1] + bbox[3]/2 )* h, xref='x', yref='y',line_color='cyan')
  fig.show()

# %%
# n
output_path = '/content/attacked_images_train_split'
csv_output_name = 'results_train_split.csv'

imgs_folder = f'LP_detection_{12}/images/train'
labels_folder = f'LP_detection_{12}/labels/train'
imgs_val_folder = 'LP_detection/images/val'
labels_val_folder = 'LP_detection/labels/val'

# %%
# n
paths = f"/content/{imgs_folder}"
img_files = sorted(os.listdir(paths))
img_path = os.path.join(paths, img_files[0])
labels = img_path.replace('images', 'labels')
labels = labels.replace('.jpg', '.txt')


input_img = cv2.imread(img_path)
cv2_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
h, w, _ = cv2_image.shape
bbox = get_bbox(labels)
visualize_bbox(cv2_image, bbox, w, h)

# %%
def prepare_bbox_for_blurring(bbox, w, h):
  # return xmin, xmax, ymin, ymax
  xmin = (int)((bbox[0] - bbox[2] / 2) * w)
  xmax = (int)((bbox[0] + bbox[2] / 2) * w)
  ymin = (int)((bbox[1] - bbox[3] / 2) * h)
  ymax = (int)((bbox[1] + bbox[3] / 2) * h)

  return xmin, xmax, ymin, ymax

# %%
# n
pixelate = Pixelate(21)
bbox = get_bbox(labels)
deid_bbox = prepare_bbox_for_blurring(bbox, w, h)
deid_img = pixelate(cv2_image.copy(), deid_bbox)

# Display the image using matplotlib
plt.imshow(deid_img)
plt.axis('off')  # Hide axis
plt.show()

# %% [markdown]
# ### Blur dataset

# %%
def blur_dataset(images_path, labels_path, deid_fn):
  deid_batch = []
  df = pd.DataFrame()
  # List the files in the directory
  images_list = sorted(os.listdir(images_path))
  labels_list = sorted(os.listdir(labels_path))
  for i, (image_file, label_file) in tqdm(enumerate(zip(images_list, labels_list)), total=len(images_list)):
    # print(image_file, " ", label_file)
    file_path = os.path.join(images_path, image_file)
    label_path = os.path.join(labels_path, label_file)
    input_img = cv2.imread(file_path)
    cv2_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    h, w, _ = cv2_image.shape

    bbox = get_bbox(label_path)
    deid_bbox = prepare_bbox_for_blurring(bbox, w, h)
    deid_img = deid_fn(cv2_image.copy(), deid_bbox)

    deid_batch.append(deid_img)

  df['file_name'] = images_list
  df['label_name'] = labels_list
  df['img'] = deid_batch
  return df

# %%
# Path to the directory
val_img_path = f"/content/{imgs_val_folder}"
val_label_path = f"/content/{labels_val_folder}"
train_img_path = f"/content/{imgs_folder}"
train_label_path = f"/content/{labels_folder}"

deid_fn = Blur(14)
# deid_fun = Pixelate(21)

val_deid_batch = blur_dataset(val_img_path, val_label_path, deid_fn)
train_deid_batch = blur_dataset(train_img_path, train_label_path, deid_fn)

# %%
# Create the folder that hold the Blur dataset
if not os.path.exists('/content/LP_detect_blur/images/val'):
    os.makedirs('/content/LP_detect_blur/images/val')

if not os.path.exists('/content/LP_detect_blur/labels/val'):
    os.makedirs('/content/LP_detect_blur/labels/val')

# Create the folder that hold the Blur dataset
if not os.path.exists('/content/LP_detect_blur/images/train'):
    os.makedirs('/content/LP_detect_blur/images/train')

if not os.path.exists('/content/LP_detect_blur/labels/train'):
    os.makedirs('/content/LP_detect_blur/labels/train')

# save validation blur images and labels
for i, (img, file_name) in tqdm(enumerate(zip(val_deid_batch['img'], val_deid_batch['file_name'])), total=len(val_deid_batch)):
    # cv2 convert back to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # save the image
    cv2.imwrite(os.path.join('/content/LP_detect_blur/images/val', file_name), img)

    # copy the label file to this directory
    label_file = file_name.replace('.jpg', '.txt')
    label_path = os.path.join(val_label_path, label_file)
    shutil.copy(label_path, os.path.join('/content/LP_detect_blur/labels/val', label_file))

# save train blur images and labels
for i, (img, file_name) in tqdm(enumerate(zip(train_deid_batch['img'], train_deid_batch['file_name'])), total=len(train_deid_batch)):
    # cv2 convert back to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # save the image
    cv2.imwrite(os.path.join('/content/LP_detect_blur/images/train', file_name), img)

    # copy the label file to this directory
    label_file = file_name.replace('.jpg', '.txt')
    label_path = os.path.join(train_label_path, label_file)
    shutil.copy(label_path, os.path.join('/content/LP_detect_blur/labels/train', label_file))

# %%
# Display the image using matplotlib
bid = 100
fig = px.imshow(val_deid_batch['img'][bid])
fig.show()

# %% [markdown]
# ## Metrics

# %% [markdown]
# ### Detection metrics

# %%
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

# %% [markdown]
# ### OCR metric

# %%
from collections import Counter

# Similarity score for Optical Character Recognition
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

# Example usage
str1 = '51A01204'
str2 = 'A0312-514'
print(similarity_metric(str1, str2))

# %%
!pip -q install levenshtein

# %%
from Levenshtein import distance
def cer_metric(ground_truth, pred):
    edit_operations = distance(ground_truth, pred)
    cer = edit_operations / len(ground_truth)
    return cer

# %% [markdown]
# ## Detect and OCR using pretrained YOLOv5 models

# %%
# Use pre-trained model and utils from "Vietnamese License Plate Recognition"
%cd /content
!git clone https://github.com/trungdinh22/License-Plate-Recognition.git

# %%
%cd /content/License-Plate-Recognition
!pip -q install -r requirement.txt

# %%
# Download YOLOv5 model from Ultralytics
!git clone https://github.com/ultralytics/yolov5.git

# %%
%cd /content/License-Plate-Recognition
from IPython.display import display
from skimage import io
from PIL import Image
import numpy as np
import shutil
import torch
import math
import cv2
import os

# %% [markdown]
# ### Prepare Models

# %%
# Run twice to ensure models work correctly
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_LP_OCR = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')

# %%
# Freeze model params
for param in yolo_LP_detect.model.model.parameters():
    param.requires_grad = False

# %%
for param in yolo_LP_OCR.model.model.parameters():
    param.requires_grad = False

# %% [markdown]
# ### Single Image Detection & OCR

# %% [markdown]
# #### Detection

# %%
# set model confidence threshold
# yolo_LP_detect.conf = 0.6
yolo_LP_OCR.conf = 0.60

# %%
# n
import function.utils_rotate as utils_rotate
import function.helper as helper
import os
# detect the plates
# paths = f"/content/{imgs_folder}"
# img_files = sorted(os.listdir(paths))
# img_file = os.path.join(paths, img_files[0])
img_file = '/content/LP_detection/images/train/xemay1541.jpg'
img = cv2.imread(img_file)
yolo_LP_detect.eval()
plates = yolo_LP_detect(img, size=640)

# OCR plates
yolo_LP_OCR.eval()
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()
count = 0
if len(list_plates) == 0:
    lp = helper.read_plate(yolo_LP_OCR,img)
    if lp != "unknown":
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        count+=1
        for cc in range(0,2):
            for ct in range(0,2):
                lp = helper.read_plate(yolo_LP_OCR, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    flag = 1
                    break
            if flag == 1:
                break

# %%
# n
paths = f"/content/{imgs_folder}"
img_files = sorted(os.listdir(paths))
file_path = os.path.join(paths, img_files[0])
# img = cv2.imread(file_path) #read the image
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# xmin-1804/ymin-1734/xmax-2493/ymax-1882
img = io.imread(file_path) #Read the image
fig = px.imshow(img)
fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Visualize detect results')

for plate in list_plates:
  fig.add_shape(type='rect',x0=plate[0], x1=plate[2], y0=plate[1], y1=plate[3], xref='x', yref='y',line_color='red')
fig.show()

# %%
# n
# detected plates bounding boxes
print(list_plates)
# ocr characters
# print(list_read_plates)

# %% [markdown]
# #### OCR

# %%
# # Function to get bounding box for each detected character
import math

# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a*x+b
    # print("distence: ", y_pred - y)
    return(math.isclose(y_pred, y, abs_tol = 3)), y_pred - y

# detect character and number in license plate
def read_plate_and_bboxes(yolo_license_plate, im):
    LP_type = "1"
    results = yolo_license_plate(im, size=640)
    bb_list = results.pandas().xyxy[0].values.tolist()
    # if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
    if len(bb_list) == 0:
        return "unknown", bb_list, 0
    center_list = []
    y_mean = 0
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0]+bb[2])/2
        y_c = (bb[1]+bb[3])/2
        y_sum += y_c
        center_list.append([x_c,y_c,bb[-1]])

    # find 2 point to draw line
    l_point = center_list[0]
    r_point = center_list[0]
    min_distance = 300
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    for ct in center_list:
        if l_point[0] != r_point[0]:
          check, distance = check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1])
          if abs(distance) < abs(min_distance):
            min_distance = abs(distance)

            # if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
          if check:
                LP_type = "2"

    y_mean = int(int(y_sum) / len(bb_list))
    size = results.pandas().s

    # 1 line plates and 2 line plates
    line_1 = []
    line_2 = []
    license_plate = ""
    # print("Lp_type:", LP_type)
    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key = lambda x: x[0]):
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key = lambda x: x[0]):
            license_plate += str(l2[2])
    else:
        for l in sorted(center_list, key = lambda x: x[0]):
            license_plate += str(l[2])
    return license_plate, bb_list, min_distance

# %%
def detect_plate_type(center_list, y_mean):
    y_variance = sum([(c[1] - y_mean) ** 2 for c in center_list]) / len(center_list)
    print(y_variance)
    threshold = 640
    if y_variance < threshold:  # Tune this threshold based on your data
        return "1"
    else:
        return "2"

def read_plate_and_bboxes(yolo_license_plate, im):
    results = yolo_license_plate(im, size=640)
    bb_list = results.pandas().xyxy[0].values.tolist()
    if len(bb_list) == 0:
        return "unknown", bb_list

    center_list = []
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, bb[-1]])

    y_mean = int(y_sum / len(bb_list))

    # Detect license plate type
    LP_type = detect_plate_type(center_list, y_mean)

    line_1 = []
    line_2 = []
    license_plate = ""

    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)

        line_1.sort(key=lambda x: (x[1], x[0]))  # Sort by y then x for line 1
        line_2.sort(key=lambda x: (x[1], x[0]))  # Sort by y then x for line 2

        for l1 in line_1:
            license_plate += str(l1[2])
        license_plate += "-"
        for l2 in line_2:
            license_plate += str(l2[2])
    else:
        center_list.sort(key=lambda x: (x[0]))  # Sort by x for single line plates
        for l in center_list:
            license_plate += str(l[2])

    return license_plate, bb_list

# %%
# n
img = cv2.imread(img_file)
list_read_plates = []
list_read_char_bbox = []
yolo_LP_OCR.eval()
if len(list_plates) == 0:
    lp, bbox, distance = read_plate_and_bboxes(yolo_LP_OCR, img)
    if lp != "unknown":
        list_read_plates.append(lp)
        list_read_char_bbox.append(bbox)
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin
        h = int(plate[3] - plate[1]) # ymax - ymin
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        cv2.imwrite("crop.jpg", crop_img)
        rc_image = cv2.imread("crop.jpg")
        lp = ""
        count+=1
        for cc in range(0,2):
            for ct in range(0,2):
                lp, bboxes, distance = read_plate_and_bboxes(yolo_LP_OCR, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.append(lp)
                    list_read_char_bbox.append(bboxes)
                    flag = 1
                    break
            if flag == 1:
                break

# %%
# n
# list này là kết quả trả về ban đầu của mô hình ocr
# mỗi phần tử của list bao gồm:
#   bounding box của ký tự: xmin, ymin, xmax, ymax
#   index của nhãn: 0 -> 36
#   độ tin cậy: score !
#   nhãn tương ứng với index: 0 -> Z
print(list_read_char_bbox)
print(distance)

# %% [markdown]
# ### Evaluate detect with YOLOv5 model

# %% [markdown]
# #### Evaluate Detect Single Image

# %%
from yolov5.utils.augmentations import letterbox

#input is normalize bbox from origin lable, xc, yc, w, h
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

# %%
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

# %%
# n
paths = f"/content/{imgs_folder}"
img_files = sorted(os.listdir(paths))
origin_image_path = os.path.join(paths, img_files[0])

origin_pred_iou = evaluate_detect_single_img(origin_image_path, verbose=True)

print(f"Origin image IoU: {origin_pred_iou}")

# %% [markdown]
# ## Detect with unauthorized model

# %%
!pip -q install tensorflow
!pip -q install pytesseract

# %%
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

# %% [markdown]
# ### Object detection h5 model

# %%
!gdown --fuzzy https://drive.google.com/file/d/1OFwE_vs0gd3eYN1wDcomzvVKZHiHAwy3/view
!unzip -q object_detection.zip
!rm object_detection.zip

# %%
inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)


# ---------- model
model = Model(inputs=inception_resnet.input,outputs=headmodel)
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# %%
model.load_weights('/content/License-Plate-Recognition/object_detection.h5')

# %%
origin_path = '/content/LP_detection/images/val/CarLongPlate105.jpg'
blur_path = '/content/LP_detect_blur/images/val/CarLongPlate105.jpg'
label_path = origin_path.replace('.jpg', '.txt')
label_path = label_path.replace('images', 'labels')

origin_image = load_img(origin_path) # PIL object
origin_image = np.array(origin_image,dtype=np.uint8) # 8 bit array (0,255)
reshape_origin_image = load_img(origin_path,target_size=(224,224))
origin_image_arr_224 = img_to_array(reshape_origin_image)/255.0  # Convert into array and get the normalized output

blur_image = load_img(blur_path) # PIL object
blur_image = np.array(blur_image,dtype=np.uint8) # 8 bit array (0,255)
reshape_blur_image = load_img(blur_path,target_size=(224,224))
blur_image_arr_224 = img_to_array(reshape_blur_image)/255.0  # Convert into array and get the normalized output


# Size of the orginal image
h,w,d = origin_image.shape
print('Height of the image =',h)
print('Width of the image =',w)
fig = px.imshow(origin_image)
fig.update_layout(width=700, height=500,  margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Test Image')

# %%
def prediction(origin_image, reshape_img_arr, h, w, verbose=False):
  # add batch dimension
  reshape_img_arr = reshape_img_arr.reshape(1,224,224,3)

  # Make predictions
  coords = model.predict(reshape_img_arr)
  # Denormalize the values
  denorm = np.array([w,w,h,h])
  coords = coords * denorm
  coords = coords.astype(np.int32)
  # Draw bounding on top the image
  xmin, xmax,ymin,ymax = coords[0]
  if verbose:
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    cv2.rectangle(origin_image,pt1,pt2,(0,255,0),3)
    fig = px.imshow(origin_image)
    fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
    fig.show()
  return coords[0]

# %%
truth_bbox = get_bbox(label_path)
visualize_bbox(origin_image, truth_bbox, w, h)

# %%
origin_pred = prediction(origin_image, origin_image_arr_224, h, w, verbose=True)
blur_pred = prediction(blur_image, blur_image_arr_224, h, w,  verbose=True)

# %%
truth_scale_bbox = prepare_bbox_for_blurring(truth_bbox, w, h)
origin_pred_iou = IoU(truth_scale_bbox, origin_pred)
blur_pred_iou = IoU(truth_scale_bbox, blur_pred)
print("origin prediction iou: ", origin_pred_iou)
print("blured prediction iou: ", blur_pred_iou)

# %% [markdown]
# ## Adversarial Attack

# %% [markdown]
# ### Prepare images and targets

# %% [markdown]
# Việc khởi tạo dataloader cho toàn bộ thư mục có điểm lợi là hỗ trợ truy xuất theo batch (load một lần nhiều hình ảnh trong cùng một batch để tăng tốc độ xử lý), nhưng hạn chế của sử dụng dataloader là cần phải tạo trước thư mục chứa tất cả các ảnh và nhãn cần load. Chính vì vậy, đối với tập mờ, cần phải làm mờ toàn bộ hình ảnh và sao chép nhãn từ nhãn của tập gốc sang cho tập mờ để build targets (điều này cũng mâu thuẩn với giả thuyết là dùng kết quả predict của mô hình trên ảnh gốc làm target, thay vì dùng nhãn làm target). Khi cần tính linh hoạt để xử lý cho từng ảnh trước khi đưa vào attack (ví dụ như crop và làm mờ ảnh biển số xe trong việc attack cho mô hình OCR, hoặc tạo target dựa trên kết quả predict trên ảnh gốc), cần có một hàm đảm nhiệm vai trò tiền xử lý trên từng hình ảnh.

# %%
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import xyxy2xywhn
from attack.deid import Blur, Pixelate

# %% [markdown]
#  Hàm này dùng được cho cả xử lý ảnh detect và OCR.
#  Hàm này có các chức năng sau
#  + Tiền xử lý hình ảnh:
#    - Input (batch): hỉnh ảnh gốc
#    - Output (batch): hình ảnh đã padding và resize về (640, 640) (đối với mô hình detect, có thể làm mờ phần biển số cho ảnh output)
#  + Tạo targets:
#    - Input (batch): hình ảnh sau khi đã tiền xử lý ở trên, mô hình tương ứng cần tạo target
#    - Output Tensor: batch_zize x (img_idx, class_index, xmin, xmax, ymin, ymax)

# %%
def preprocess(imgs, model, deid_fn, img_size=640, need_blurred=False, blur_whole=False):
  model.eval()
  preprocess_imgs = []
  bboxes_out = []
  targets_out = []
  for img in imgs:
    # Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding.
    pad_img, ratio, pad = letterbox(img, img_size, auto=False, scaleup=True)
    p_w, p_h = pad_img.shape[1], pad_img.shape[0]

    pred = model(img)
    boxes = pred.pandas().xyxy[0].values.tolist()
    boxes = np.array([[item[5], item[0], item[1], item[2], item[3]] for item in boxes])

    # return the original bboxes from origin images
    if len(boxes) == 0:
      boxes = np.zeros((0, 6))
      bboxes_out.append(boxes.copy())
    else:
      bboxes_out.append(boxes.copy())
    nl = len(boxes)
    target = torch.zeros((nl, 6))
    if nl:
      # normalized [x1, y1, x2, y2] to size after padding
      boxes[..., 1] = boxes[..., 1] * ratio[0] + pad[0] # top left x
      boxes[..., 2] = boxes[..., 2] * ratio[1] + pad[1] # top left y
      boxes[..., 3] = boxes[..., 3] * ratio[0] + pad[0] # bottom right x
      boxes[..., 4] = boxes[..., 4] * ratio[1] + pad[1]  # bottom right y
      if need_blurred:
        if blur_whole:
          deid_fn = Blur(int(8 * ratio[0]))
          # deid_fn = Pixelate(int(21 * ratio[0]))
          pad_img = deid_fn(pad_img.copy(), (0, img_size, 0, img_size))
        else:
          for box in boxes:
            x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
            pad_img = deid_fn(pad_img.copy(), (int(x1), int(x2), int(y1), int(y2)))

      # convert xyxy to xc, yc, wh
      boxes[:, 1:5] = xyxy2xywhn(boxes[:, 1:5], w=p_w, h=p_h, clip=True, eps=1e-3)
      target[:, 1:] = torch.from_numpy(boxes)

    pad_img = pad_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    pad_img = np.ascontiguousarray(pad_img)
    preprocess_imgs.append(pad_img)
    targets_out.append(target)

  preprocess_imgs = np.stack(preprocess_imgs, axis=0)
  targets_out = np.stack(targets_out, axis=0)
  return torch.from_numpy(preprocess_imgs), torch.from_numpy(targets_out), bboxes_out

# %% [markdown]
# ### Testing the preprocess method

# %%
paths = f"/content/{imgs_folder}"
img_files = sorted(os.listdir(paths))
test_img_path_1 = os.path.join(paths, img_files[0])
test_img_path_2 = os.path.join(paths, img_files[1])
test_imgs = [cv2.imread(test_img_path_1), cv2.imread(test_img_path_2)]

deid_fn = Blur(10)
preprocess_imgs, targets, bboxes_list = preprocess(test_imgs, yolo_LP_detect, deid_fn, img_size=640, need_blurred=True)

bid = 0
print(preprocess_imgs[bid].shape, type(preprocess_imgs[bid]))
image =  preprocess_imgs[bid].numpy()
print(image.shape, type(image))
# Transpose the dimensions to [640, 640, 3]
image = np.transpose(image, (1, 2, 0))
# Plot the image using Plotly Express
fig2 = px.imshow(image)
fig2.show()
print(targets, type(targets), targets.shape)

# %%
bbox = targets[bid].squeeze(0).numpy()[2:6]
visualize_bbox(image, bbox, 640, 640)

# %%
# crop on original images to build right target
# ocr_inputs is a list of multible of lists of images, each list of images is the list of plates detected in an input image
# if no plate is detected, the whole input image is used for an item of ocr_inputs, notice that the whole image is contained in a list for compatability
# len(ocr_inputs) = len(batch_imgs)
def crop_batch_imgs_and_build_ocr_targets(batch_imgs, bboxes_list, ocr_model, return_ocr_preprocess = False, deid_fn=Blur(14)):
  crop_plates_batch = []
  for i, (img, bboxes)  in enumerate(zip(batch_imgs, bboxes_list)):
    # check if bboxes contain zeros
    crop_imgs = []
    if bboxes.shape[0]:
      for bbox in bboxes:
        x = int(bbox[1]) # xmin
        y = int(bbox[2]) # ymin
        w = int(bbox[3] - bbox[1]) # xmax - xmin
        h = int(bbox[4] - bbox[2]) # ymax - ymin
        crop_img = img[y:y+h, x:x+w]
        crop_imgs.append(crop_img.copy())
    else:
      crop_imgs.append(img)
    crop_plates_batch.append(crop_imgs)

  # testing the ocr target
  ocr_targets_batch = []
  ocr_preprocess_batch = []
  for ocr_plates in crop_plates_batch:
    ocr_preprocess_imgs, ocr_targets, bboxes_list = preprocess(ocr_plates, ocr_model, deid_fn, img_size=640, need_blurred=True, blur_whole=True)
    ocr_targets_batch.append(ocr_targets)
    ocr_preprocess_batch.append(ocr_preprocess_imgs)
  if return_ocr_preprocess:
    return  ocr_targets_batch, ocr_preprocess_batch
  else:
    return ocr_targets_batch

ocr_targets_batch = crop_batch_imgs_and_build_ocr_targets(test_imgs, bboxes_list, yolo_LP_OCR)

print(ocr_targets_batch[0].shape)
print(ocr_targets_batch[0][0])

# %% [markdown]
# ### Custom Optimization Algorithms for Iterative attack

# %%
from torch.optim.optimizer import Optimizer
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD

class I_FGSM:
    def __init__(self, params, epsilon=8/255., alpha=1/255., min_value=0, max_value=1):
        self.params = params
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_value = min_value
        self.max_value = max_value
        self.updated_params = []
        for param in self.params:
            self.updated_params.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        return -self.alpha * torch.sign(self.params[idx].grad)

    @torch.no_grad()
    def step(self):
        for idx, (param, updated_param) in enumerate(zip(self.params, self.updated_params)):
            if param is None:
                continue

            n_update = torch.clamp(updated_param + self._cal_update(idx), -self.epsilon, self.epsilon)
            update = n_update - updated_param
            n_param = torch.clamp(param + update, self.min_value, self.max_value)
            update = n_param - param

            param += update
            updated_param += update

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class MI_FGSM(I_FGSM):
    def __init__(self, params, epsilon=8/255., momemtum=0, **kwargs):
        super(MI_FGSM, self).__init__(params, epsilon, **kwargs)
        self.momentum = momemtum
        self.o_grad = []
        for param in self.params:
            self.o_grad.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        grad = self.o_grad[idx] * self.momentum + self.params[idx].grad / torch.sum(torch.abs(self.params[idx].grad))
        return -self.alpha * torch.sign(grad)

    def zero_grad(self):
        for o_grad, param in zip(self.o_grad, self.params):
            if param.grad is not None:
                o_grad = o_grad * self.momentum + param.grad / torch.sum(torch.abs(param.grad))
        super().zero_grad()

class WrapOptim:
    @torch.no_grad()
    def __init__(self, params, epsilon, optimizer:Optimizer, min_value=0, max_value=1):
        self.optim = optimizer
        self.params = params
        self.epsilon = epsilon
        self.min_value = min_value
        self.max_value = max_value
        self.params_init = []
        for param in params:
            self.params_init.append(param.clone())

    @torch.no_grad()
    def step(self):
        self.optim.step()
        for param, param_init in zip(self.params, self.params_init):
            total_update = param - param_init
            update = torch.clamp(total_update, -self.epsilon, self.epsilon)

            param += update - total_update
            param.clamp_(self.min_value, self.max_value)

    def zero_grad(self):
        self.optim.zero_grad()


def get_optim(name, params, epsilon, **kwargs) -> I_FGSM:
    if name == 'i-fgsm':
        return I_FGSM(params, epsilon, **kwargs)
    if name == 'mi-fgsm':
        return MI_FGSM(params, epsilon, **kwargs)

    optimizer = None
    if name == 'adadelta':
        optimizer = Adadelta(params)
    if name == 'adagrad':
        optimizer = Adagrad(params)
    if name == 'adam':
        optimizer = Adam(params)
    if name == 'adamw':
        optimizer = AdamW(params)
    if name == 'adamax':
        optimizer = Adamax(params)
    if name == 'asgd':
        optimizer = ASGD(params)
    if name == 'rmsprop':
        optimizer = RMSprop(params, lr=0.004)
    if name == 'rprop':
        optimizer = Rprop(params)
    if name == 'sgd':
        optimizer = SGD(params)

    if optimizer:
        return WrapOptim(params, epsilon, optimizer, **kwargs)

    return None

# %% [markdown]
# ### Attack

# %% [markdown]
# Trước khi chạy attack, cần phải chạy:
# - Prapare images and targets
# - Custom Optimization
# - detect_dataset, evaluate_detect_for_yolo, prepare_bbox_afer_attack trong phần detect và eveluate trên dataset và trên single image

# %% [markdown]
# #### Attack Single Use preprocess method

# %%
import os
import torch.nn.functional as F
from yolov5.utils.loss import ComputeLoss
from tqdm.auto import tqdm

# %%
import os
from yolov5.utils.loss import ComputeLoss

if not os.path.exists('/content/attack_ocr_detect_results'):
    os.mkdir('/content/attack_ocr_detect_results')
    os.mkdir('/content/attack_ocr_detect_results/images')
    os.mkdir('/content/attack_ocr_detect_results/images/train')
    os.mkdir('/content/attack_ocr_detect_results/images/val')

# %%
img_files = ['/content/LP_detection/images/train/CarLongPlate102.jpg', '/content/LP_detection/images/train/xemay162.jpg', '/content/LP_detection/images/train/xemay1589.jpg']

batch_imgs = []
for img_file in img_files:
    img = cv2.imread(img_file)
    batch_imgs.append(img)

print(len(batch_imgs))

# %% [markdown]
# Tiền xử lý hình ảnh gốc, kết quả trả về là hình ảnh đã resize, padding, và làm mờ cho nhiệm vụ attack detect.

# %%
deid_fn = Blur(14)
preprocess_imgs, detect_targets, bboxes_list = preprocess(batch_imgs, yolo_LP_detect, deid_fn, img_size=640, need_blurred=True)

bid=0
image = preprocess_imgs[1].numpy()
print(image.shape, type(image))
image = np.transpose(image, (1, 2, 0))
# Plot the image using Plotly Express
fig2 = px.imshow(image)
fig2.show()
print(preprocess_imgs.shape)
print("detect target shape: ", detect_targets.shape)

# %% [markdown]
# Hàm sau đây nhận vào batch images là những hình ảnh gốc, và danh sách vị trí các biển số có trong từng hình ảnh gốc. Hàm này sẽ crop mỗi hình ảnh gốc thành danh sách các hình ảnh biển số có trong ảnh gốc. Sau đó, hàm preprocess tiếp tục được gọi, lần này là nhận vào các hình ảnh biển số xe đã được crop và tính ra ocr target trên các ảnh biển số gốc này. Lưu ý là hàm này chỉ tạo ra kết quả cần dùng là ocr targets, chứ không thực hiện việc crop phần biển số đã làm mờ và tiền xử lý (padding, resize) để tạo ra input cho attack ocr.
# 
# Hàm này là một phần của quá trình preprocess, diễn ra trước khi đi vào vòng lặp attack. Mục tiêu của hàm là tính ra ocr targets.

# %%
ocr_targets_batch = crop_batch_imgs_and_build_ocr_targets(batch_imgs, bboxes_list, yolo_LP_OCR)

print(ocr_targets_batch[0].shape)
print(ocr_targets_batch[0][0])

# %% [markdown]
# Hàm này thực hiện crop vùng biển số đã làm mờ từ hình ảnh đã được preprocess lần đầu tiên cho nhiệm vụ detect. Hàm này cũng thực hiện padding và resize hình ảnh đã crop, tạo ra input cho việc attack ocr. Hàm này được gọi bên trong vòng lặp attack. Input của hàm sẽ thay đổi trong quá trình cập nhật hình ảnh trong lúc attack.

# %%
core_model_detect = yolo_LP_detect.model.model
core_model_OCR = yolo_LP_OCR.model.model
compute_loss_detect = ComputeLoss(core_model_detect)
compute_loss_OCR = ComputeLoss(core_model_OCR)

# Điều chỉnh việc chia epsilon để tăng hiệu suất (chạy lâu hơn)
eps1 = 0.022834776253708135 / 2
eps2 = 0.043090941003692 / 5
batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ITER_NUM = 300

# %%
attacked_imgs = []

for img, detect_target, ocr_targets in zip(preprocess_imgs, detect_targets, ocr_targets_batch):

    img = img.unsqueeze(0).to(device).float() / 255.0
    ocr_targets = ocr_targets.squeeze(0).to(device)
    detect_target = detect_target.to(device)
    img.requires_grad = True

    loss_detect_list = []
    loss_ocr_list = []
    loss_TB_list = []

    core_model_detect.train()
    core_model_OCR.train()
    optim = get_optim('rmsprop', params=[img], epsilon=8 / 255.0)

    iter_cnt = 0
    while True:
        optim.zero_grad()
        with torch.set_grad_enabled(True):
            det_pred = core_model_detect(img)
            det_loss, det_loss_items = compute_loss_detect(det_pred, detect_target)

            # crop input with plate area
            # ocr_attack_inputs = crop_tensor(img.squeeze(0), detect_target)
            # ocr_pred = core_model_OCR(ocr_attack_inputs)
            # ocr_batch_size = ocr_attack_inputs.shape[0]

            # use the whole image as ocr input
            ocr_pred = core_model_OCR(img)
            ocr_loss, ocr_loss_items = compute_loss_OCR(ocr_pred, ocr_targets)

            if det_loss.item() / batch_size > eps1:
              loss = det_loss
              loss_detect_list.append(det_loss.item())
            # elif ocr_loss.item() / ocr_batch_size > eps2:
            elif ocr_loss.item() / batch_size > eps2:
              loss = ocr_loss + det_loss
              loss_ocr_list.append(ocr_loss.item())
              loss_TB_list.append(ocr_loss.item() + det_loss.item())
            else:
              break

            # print(loss.item())
            loss.backward()

        optim.step()

        iter_cnt += 1
        if iter_cnt == ITER_NUM:
            break
    attacked_imgs.append(img.detach().cpu().numpy().transpose((0, 2, 3, 1)).squeeze(0) * 255.0)

# %%
for i, img in enumerate(attacked_imgs):
    filename = os.path.basename(img_files[i])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('/content/attack_ocr_detect_results/images/train', filename), img)

# %%
# visuallize loss items using px
fig = px.line(y=loss_detect_list, title='Loss detect during attack detect')
fig.show()

# visuallize loss items using px
fig = px.line(y=loss_ocr_list, title='Loss OCR during attack detect')
fig.show()

# visuallize loss items using px
fig = px.line(y=loss_TB_list, title='Loss TB during attack detect')
fig.show()

# %%
%cd /content/License-Plate-Recognition
import function.utils_rotate as utils_rotate
import function.helper as helper

import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_images(img_path1, img_path2, plates, ocr_bboxes):
    # Đọc ảnh từ đường dẫn
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # Chuyển đổi ảnh từ BGR sang RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Tạo subplot với 1 hàng và 2 cột
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image 1", "Image 2"))

    # Thêm ảnh vào subplot
    fig.add_trace(go.Image(z=img1), row=1, col=1)
    fig.add_shape(type='rect',x0=plates[0][0], x1=plates[0][2], y0=plates[0][1], y1=plates[0][3], xref='x', yref='y',line_color='red', row=1, col=1)
    for bbox in ocr_bboxes:
        fig.add_shape(type='rect',x0=bbox[0], x1=bbox[2], y0=bbox[1], y1=bbox[3], xref='x', yref='y',line_color='yellow', row=1, col=1)
        fig.add_annotation(
            x=(bbox[0] + bbox[2]) / 2,  # x coordinate for the label (center of the rectangle)
            y=bbox[3],  # y coordinate for the label (top of the rectangle)
            text=bbox[6],  # The label text
            showarrow=False,  # No arrow pointing to the label
            font=dict(size=12, color="yellow"),  # Label font and color
            xref="x", yref="y",
            xanchor="center",  # Anchor the label horizontally at the center
            yanchor="bottom"  # Anchor the label vertically at the bottom
        )

    fig.add_trace(go.Image(z=img2), row=1, col=2)

    # Cập nhật layout
    fig.update_layout(height=1000, width=1200, title_text="Two Images Side by Side")

    # Hiển thị biểu đồ
    fig.show()

att_img_path = '/content/attacked_images/attacked_imges_train_split_12/xemay1541.jpg'
original_path = '/content/LP_detection/images/train/xemay1541.jpg'

att_img = cv2.imread(att_img_path)
origin_img = cv2.imread(original_path)
# fig = px.imshow(cv2.cvtColor(att_img, cv2.COLOR_BGR2RGB))

yolo_LP_detect.eval()
pred = yolo_LP_detect(att_img, size=640)
plates = pred.pandas().xyxy[0].values.tolist()

print("plate detect: ", plates)

yolo_LP_OCR.eval()
lp, bboxes = read_plate_and_bboxes(yolo_LP_OCR, att_img)

display_images(att_img_path, original_path, plates, bboxes)

for bbox in bboxes:
    print(bbox)
print("ocr: ", lp)

# %%
from tqdm.auto import tqdm

files = os.listdir('/content/attacked_images/attacked_imges_train_split_12')
min_distance = 300
max_file = ""
for file in tqdm(files, total = len(files)):
    # print(file)
    if file == 'train' or file == 'val':
      continue
    att_img_path = os.path.join('/content/attacked_images/attacked_imges_train_split_12', file)
    origin_path = os.path.join('/content/LP_detection/images/train', file)

    att_img = cv2.imread(att_img_path)
    img = cv2.imread(origin_path)

    # lp, bboxes, distance = read_plate_and_bboxes(yolo_LP_OCR, att_img)
    yolo_LP_detect.eval()
    pred = yolo_LP_detect(img, size=640)
    list_plates = pred.pandas().xyxy[0].values.tolist()
    yolo_LP_OCR.eval()
    if len(list_plates) == 0:
        lp, bbox, distance = read_plate_and_bboxes(yolo_LP_OCR, img)
        distance = 300
        if lp == 'unknown':
            continue
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin
            crop_img = img[y:y+h, x:x+w]
            for cc in range(0,2):
                for ct in range(0,2):
                    lp, bboxes, distance = read_plate_and_bboxes(yolo_LP_OCR, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != 'unknown':
                        flag = 1
                        break
                    else:
                      distance = 300
                if flag == 1:
                    break
    if distance < min_distance:
        min_distance = distance
        max_file = file

print(min_distance)
print(max_file)

# %% [markdown]
# ### Eval

# %%
%cd /content/License-Plate-Recognition
import function.utils_rotate as utils_rotate
import function.helper as helper
origin_paths = f"/content/{imgs_folder}"
attacked_paths = '/content/attacked_imges_train_split_12'

yolo_LP_detect = yolo_LP_detect.eval()
yolo_license_plate = yolo_LP_OCR.eval()

origin_files = sorted(os.listdir(origin_paths))
attacked_files = sorted(os.listdir(attacked_paths))

origin_det_preds = []
det_ious = []
att_det_preds =[]
origin_ocr_preds = []
att_ocr_preds = []
ocr_similarities = []
ocr_cer = []
detectable_file_names = []
for i, (img_file) in enumerate(tqdm(origin_files, total=len(origin_files))):
    att_img_file = os.path.join(attacked_paths, img_file)
    img_path = os.path.join(origin_paths, img_file)
    att_img_path = os.path.join(attacked_paths, att_img_file)
    origin_img = cv2.imread(img_path)
    h, w, _ = origin_img.shape
    H, W = (640, 640)
    ratio = W / w
    pad_y = (H - h * ratio) / 2
    att_img = cv2.imread(att_img_path)

    origin_pred = yolo_LP_detect(origin_img, size=640)
    origin_plates = origin_pred.pandas().xyxy[0].values.tolist()
    origin_bbox = [0, 0, 0, 0]
    if len(origin_plates):
        origin_bbox = [
            origin_plates[0][0] * ratio,
            origin_plates[0][2] * ratio,
            origin_plates[0][1] * ratio + pad_y,
            origin_plates[0][3] * ratio + pad_y
        ]
    else:
        continue

    detectable_file_names.append(img_file)

    if len(origin_plates):
        flag = 0
        x = int(origin_plates[0][0]) # xmin
        y = int(origin_plates[0][1]) # ymin
        w = int(origin_plates[0][2] - origin_plates[0][0]) # xmax - xmin
        h = int(origin_plates[0][3] - origin_plates[0][1]) # ymax - ymin
        crop_img = origin_img[y:y+h, x:x+w]
        flag = 0
        for cc in range(0,2):
            for ct in range(0,2):
                origin_lp, bboxes = read_plate_and_bboxes(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if origin_lp != "unknown":
                    origin_ocr_preds.append(origin_lp)
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            origin_ocr_preds.append("unknown")
    else:
        origin_ocr_preds.append("unknown")

    att_pred = yolo_LP_detect(att_img, size=640)
    att_plates = att_pred.pandas().xyxy[0].values.tolist()
    att_bbox = [0, 0, 0, 0]
    if len(att_plates):
        att_bbox = [att_plates[0][0], att_plates[0][2], att_plates[0][1], att_plates[0][3]]
    att_lp, bboxes = read_plate_and_bboxes(yolo_license_plate, att_img)

    origin_det_preds.append(origin_bbox)
    att_det_preds.append(att_bbox)
    att_ocr_preds.append(att_lp)
    det_ious.append(IoU(origin_bbox, att_bbox))
    ocr_similarities.append(similarity_metric(origin_lp, att_lp))
    ocr_cer.append(cer_metric(origin_lp, att_lp))


result = pd.DataFrame({
    'file_path': detectable_file_names,
    'origin detection': origin_det_preds,
    'attacked detection': att_det_preds,
    'iou': det_ious,
    'origin OCR': origin_ocr_preds,
    'attacked OCR': att_ocr_preds,
    'OCR similarity': ocr_similarities,
    'OCR CER': ocr_cer
  })

# %%
result.to_csv('/content/attacked_images_train_split_12.csv')


