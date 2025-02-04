from collections import Counter
from Levenshtein import distance
import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.append("../")

import cv2
from _models.detection.YoLoDetect import YOLOv5Detector
from _models.OCR.YoloOCR import YoloLicensePlateOCR
# from _models.OCR.easyOCR import EasyOCR
# from _models.detection.InceptionResnet import InceptionResnet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate OCR and Detection Models")
    parser.add_argument("--detection_model", type=str, required=False, default="yolo", help="Path to the detection model")
    parser.add_argument("--ocr_model", type=str, required=False, default="yolo", help="Path to the OCR model")
    parser.add_argument("--origin_images_path", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--adv_images_path", type=str, required=True, help="Path to the folder containing adversarial images")
    parser.add_argument("--batch_size", type=int, required=False, default=1, help="Batch size")
    return parser.parse_args()

class Evaluate:
    def __init__(self, model_detection, model_ocr, origin_images_path, adv_images_path, batch_size):
        self.model_detection = model_detection
        self.model_ocr = model_ocr
        self.origin_images_path = origin_images_path
        self.adv_images_path = adv_images_path
        self.batch_size = batch_size
        self.origin_images = None
        self.adv_images = None
        self.detection_result_before_deid = None
        self.detection_result_after_attack = None
        self.ocr_result_before_deid = None
        self.ocr_result_after_attack = None

    def load_images(self):
        """Load images from folder"""

        origin_dir = os.listdir(self.origin_images_path)
        adv_dir = os.listdir(self.adv_images_path)

        self.cv2_images = []
        self.adv_images = []

        if len(origin_dir) > 0:
            self.origin_images = [cv2.imread(os.path.join(self.origin_images_path, file)) for file in origin_dir]
        else:
            print("No images found in the origin folder. Exiting...")
            sys.exit(1)

        if len(adv_dir) > 0:
            self.adv_images = [cv2.imread(os.path.join(self.adv_images_path, file)) for file in adv_dir]
        else:
            print("No images found in the adv folder. Exiting...")
            sys.exit(1)

    def IoU(self, truth_bbox, pred_bbox):
        """Calculate Intersection over Union (IoU)"""
        x1_true, y1_true, x2_true, y2_true = truth_bbox
        x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox

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

        iou = interArea / unionArea if unionArea > 0 else 0
        return iou

    def similarity_metric(self, str1, str2):
        """Calculate character similarity metric"""
        str1 = ''.join(filter(str.isalnum, str1)).upper()
        str2 = ''.join(filter(str.isalnum, str2)).upper()

        count_str1 = Counter(str1)
        count_str2 = Counter(str2)

        match_count = sum(min(count_str1[char], count_str2[char]) for char in count_str2)

        similarity = match_count / len(str1) if len(str1) > 0 else 0
        return similarity

    def cer_metric(self, ground_truth, pred):
        """Calculate Character Error Rate (CER)"""
        edit_operations = distance(ground_truth, pred)
        cer = edit_operations / len(ground_truth) if len(ground_truth) > 0 else 0
        return cer

    def inference_before_attack(self, batch_origin_images):
        """Inference detection and OCR models before attack"""
        images = self.model_detection.preprocess(batch_origin_images)
        det_predictions = self.model_detection.detect(images)
        bboxes = self.model_detection.get_bboxes(det_predictions)

        crop_images = self.model_ocr.preprocess(images, bboxes)
        rec_predictions = self.model_ocr.detect(crop_images)
        lps, _ = self.model_ocr.get_plates_and_bboxes(rec_predictions)

        return bboxes, lps
    
    def inference_after_attack(self, batch_adv_images):
        """Inference detection and OCR models after attack"""
        adv_imgs = self.model_detection.preprocess(batch_adv_images)
        det_predictions = self.model_detection.detect(adv_imgs)
        bboxes = self.model_detection.get_bboxes(det_predictions)

        rec_predictions = self.model_ocr.detect(adv_imgs)
        lps, _ = self.model_ocr.get_plates_and_bboxes(rec_predictions)
    
        return bboxes, lps
    
    def run_and_eval(self):
        """Run evaluation pipeline and print results"""
        result_df = pd.DataFrame(columns=["IoU", "Similarity Metric", "Character Error Rate (CER)"])
        self.load_images()

        for i in range(0, len(self.origin_images), self.batch_size):
            batch_origin_images = self.origin_images[i:i+self.batch_size]
            batch_adv_images = self.adv_images[i:i+self.batch_size]

            self.detection_result_before_deid, self.ocr_result_before_deid = self.inference_before_attack(batch_origin_images)
            self.detection_result_after_attack, self.ocr_result_after_attack = self.inference_after_attack(batch_adv_images)

            for (ground_truth_bbox, ground_truth_text, pred_bbox, pred_text) in zip(
                self.detection_result_before_deid, 
                self.ocr_result_before_deid, 
                self.detection_result_after_attack, 
                self.ocr_result_after_attack):
                
                iou = self.IoU(ground_truth_bbox[0], pred_bbox[0])
                similarity = self.similarity_metric(ground_truth_text, pred_text)
                cer = self.cer_metric(ground_truth_text, pred_text)

                result_df = result_df._append({"IoU": iou, "Similarity Metric": similarity, "Character Error Rate (CER)": cer}, ignore_index=True)
        result_df.to_csv("evaluation_results.csv", index=False)


    def forward(self, ground_truth_detection, ground_truth_ocr, pred_detection, pred_ocr):
        """Run evaluation pipeline and print results"""
        ious, similarities, cers = [], [], []

        for (ground_truth_bbox, ground_truth_text, pred_bbox, pred_text) in zip(ground_truth_detection, ground_truth_ocr, pred_detection, pred_ocr):
            iou = self.IoU(ground_truth_bbox, pred_bbox)
            similarity = self.similarity_metric(ground_truth_text, pred_text)
            cer = self.cer_metric(ground_truth_text, pred_text)

            ious.append(iou)
            similarities.append(similarity)
            cers.append(cer)

        # Calculate average metrics
        mean_iou = np.mean(ious)
        mean_similarity = np.mean(similarities)
        mean_cer = np.mean(cers)

        # Print results
        print("Mean evaluation Results:")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"Mean Similarity Metric: {mean_similarity:.4f}")
        print(f"Mean Character Error Rate (CER): {mean_cer:.4f}")

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

    # Initialize evaluator
    evaluator = Evaluate(
        model_detection=model_detection,
        model_ocr=model_ocr,
        origin_images_path=args.origin_images_path,
        adv_images_path=args.adv_images_path,
        batch_size=args.batch_size  
    )

    # Run evaluation
    evaluator.run_and_eval()