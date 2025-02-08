from ._models.EasyOCR import easyocr
from .base import BaseOCR
import cv2
import torch


def crop_image(image, bbox):
  xmin, ymin, xmax, ymax = bbox
  return image[ymin:ymax, xmin:xmax]

def resize_and_padding(image, img_size=640):
  h, w, _ = image.shape
  # padding and resize to imgsize

  if h > w:
    new_height = img_size
    new_width = img_size * w // h
    pad_img = cv2.resize(image, (new_width, new_height))

    if new_width < img_size:
      # padding left and right
      left_pad = (img_size - new_width) // 2
      right_pad = img_size - new_width - left_pad
      pad_img = cv2.copyMakeBorder(pad_img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

  new_width = img_size
  new_height = img_size * h // w
  pad_img = cv2.resize(image, (new_width, new_height))

  if new_height < img_size:
    # padding top and bottom
    top_pad = (img_size - new_height) // 2
    bottom_pad = img_size - new_height - top_pad
    pad_img = cv2.copyMakeBorder(pad_img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

  return pad_img

class EasyOCR(BaseOCR):
  def __init__(self):
    super(EasyOCR, self).__init__()

    self.reader = easyocr.Reader(['en'])
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.reader.recognizer.to(self.device)
    self.criterion = torch.nn.CTCLoss(zero_infinity=True).to(self.device)


  def preprocess(self, images, bboxes, img_size=640):
    preprocess_imgs = []
    for img, bbox in zip(images, bboxes):
      crop_img = crop_image(img, bbox)
      pad_img = resize_and_padding(crop_img, img_size)
      preprocess_imgs.append(pad_img)
    return preprocess_imgs

  def postprocess(self, adv_images):
    adv_images = adv_images.detach().cpu().numpy().transpose(1,2,0) * 255.0
    adv_images = cv2.cvtColor(adv_images, cv2.COLOR_RGB2BGR)
    return adv_images

  def forward(self, adv_images, targets):
    self.model.train()

    pass

  def detect(self, images):

    predictions = self.reader.readtext_batched(images)

    return predictions

  def make_targets(self, predictions, images):
    pass


  def get_plates_and_bboxes(self, predictions):
    lps = []
    bboxes = []
    for pred in predictions:
      print(pred)
      lp = ""
      bbox_list = []
      for bbox, l, _ in pred:
        bbox = [int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])]
        bbox_list.append(bbox)
        lp += l
      lps.append(lp.replace(" ", ""))
      bboxes.append(bbox_list)
    return lps, bboxes
