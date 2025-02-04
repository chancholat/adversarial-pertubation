import os
import sys

sys.path.append("../")

from _models.detection.YoLoDetect import YOLOv5Detector
from _models.OCR.easyOCR import EasyOCR
from _models.OCR.YoloOCR import YoloLicensePlateOCR
from _models.detection.InceptionResnet import InceptionResnet

from attack.attacker.full_attacker import FullAttacker
from attack.deid import Blur
import argparse

import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate OCR and Detection Models")
    parser.add_argument("--detection_model", type=str, required=False, default="yolo", help="Path to the detection model")
    parser.add_argument("--ocr_model", type=str, required=False, default="easyocr", help="Path to the OCR model")
    parser.add_argument("--images", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--optim", type=str, required=False, default="rmsprop", help="Optimizer to use")
    parser.add_argument("--max_iter", type=int, required=False, default=250, help="Max number of iterations")
    parser.add_argument("--eps", type=float, required=False, default=12, help="Epsilon value") # /255 below
    parser.add_argument("--eps1", type=float, required=False, default=0.022834776253708135, help="Epsilon value 1")
    parser.add_argument("--eps2", type=float, required=False, default=0.043090941003692 / 5.0, help="Epsilon value 2")
    parser.add_argument("--blur", type=int, required=False, default=14, help="Blur value")
    parser.add_argument("--batch_size", type=int, required=False, default=1, help="Batch size")
    return parser.parse_args()


class InferAttack:
    def __init__(self, detection_model, ocr_model, images_path, optim, max_iter, eps, eps1, eps2, blur):
        self.detection_model = detection_model
        self.ocr_model = ocr_model
        self.images_path = images_path
        self.optim = optim
        self.max_iter = max_iter
        self.eps = eps / 255.0
        self.eps1 = eps1
        self.eps2 = eps2
        self.blur = blur
        self.cv2_images = None
        self.deid_fn = Blur(blur)
        self.attacker = None

    def load_images(self):
        """Load images from folder"""
        cv2_images = []
        dir = os.listdir(self.images_path)
        if len(dir) > 0:
            for file in dir:
                img = cv2.imread(os.path.join(self.images_path, file))
                cv2_images.append(img)
            return cv2_images
        else:
            print("No images found in the folder. Exiting...")
            sys.exit(1)

    
    def deid(self, batch_images):
        images = self.detection_model.preprocess(batch_images)
        det_predictions = self.detection_model.detect(images)
        bboxes = self.detection_model.get_bboxes(det_predictions)
        
        adv_imgs = self.deid_fn.forward_batch(images, bboxes)
        return adv_imgs

    def attack(self):
        self.cv2_images = self.load_images()
        self.attacker = FullAttacker(
            self.optim, 
            self.max_iter, 
            self.eps, 
            self.eps1, 
            self.eps2
        )
        # get batch_images from self.cv2_images with batch_size = self.batch_size
        for i in range(0, len(self.cv2_images), self.batch_size):
            batch_images = self.cv2_images[i:i+self.batch_size]
            deid_images = self.deid(batch_images)

            adv_imgs = self.attacker.attack(
                victims={
                    "detection": self.detection_model,
                    "OCR": self.ocr_model
                },
                images=batch_images,
                deid_images=deid_images
            )
            # save adv_imgs using cv2
            for j, adv_img in enumerate(adv_imgs):
                cv2.imwrite(f"E:/study 7/PAPERS/SNCS/output_images/adv_img_{i}_{j}.jpg", adv_img)


if __name__ == "__main__":
    args = parse_arguments()

    
    if args.detection_model == "inception":
        model_detection = InceptionResnet()
    elif args.detection_model == "yolo":
        model_detection = YOLOv5Detector()
    
    if args.ocr_model == "yolo":
        model_ocr = YoloLicensePlateOCR()
    elif args.ocr_model == "easyocr":
        model_ocr = EasyOCR()


    infer_attack = InferAttack(
        detection_model=args.detection_model,
        ocr_model=args.ocr_model,
        images_path=args.images,
        optim=args.optim,
        max_iter=args.max_iter,
        eps=args.eps,
        eps1=args.eps1,
        eps2=args.eps2,
        blur=args.blur
    )
    
    infer_attack.attack()    