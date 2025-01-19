from collections import Counter
from Levenshtein import distance
import numpy as np
import pandas as pd
import argparse
import os
import sys

from models.detection.YoLoDetect import YoloDetection
from models.OCR.easyOCR import EasyOCR
from models.OCR.YoloOCR import YoloLicensePlateOCR
from models.detection.InceptionResnet import InceptionResnet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate OCR and Detection Models")
    parser.add_argument("--detection_model", type=str, required=False, default="yolo", help="Path to the detection model")
    parser.add_argument("--ocr_model", type=str, required=False, default="easyocr", help="Path to the OCR model")
    parser.add_argument("--images", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--ground_truth_detection_path", type=str, required=True, help="Path to ground truth detection annotations")
    parser.add_argument("--ground_truth_ocr_path", type=str, required=True, help="Path to ground truth OCR annotations")
    return parser.parse_args()

class Evaluate:
    def __init__(self, model_detection, model_ocr, images_path, ground_truth_detection_path, ground_truth_ocr_path):
        self.model_detection = model_detection
        self.model_ocr = model_ocr
        self.images_path = images_path
        self.ground_truth_detection_path = ground_truth_detection_path
        self.ground_truth_ocr_path = ground_truth_ocr_path
        self.ground_truth_detection = None
        self.ground_truth_ocr = None

    def load_images(self):
        """Load images from folder"""
        # input: self.images_path
        # output: list of image paths
        # example: ["path/to/image1.jpg", "path/to/image2.jpg", ...]

        dir = os.listdir(self.images_path)
        if len(dir) > 0:
            return [os.path.join(self.images_path, file) for file in dir]
        else:
            print("No images found in the folder. Exiting...")
            sys.exit(1)

    def load_ground_truth_detection(self):
        """Load ground truth detection annotations from txt or csv file"""
        # input: self.ground_truth_detection_path
        # output: list of ground truth detection annotations
        # example: [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
        return []
    
    def load_ground_truth_ocr(self):
        """Load ground truth OCR annotations from txt or csv file"""
        # input: self.ground_truth_ocr_path
        # output: list of ground truth OCR annotations
        # example: ["ABCD-1", "CDE-G2", ...]
        return []
    
    def load_images_and_ground_truth(self):
        """Load ground truth annotations"""
        self.images = self.load_images()
        self.ground_truth_detection = self.load_ground_truth_detection()
        self.ground_truth_ocr = self.load_ground_truth_ocr()

    def IoU(self, truth_bbox, pred_bbox):
        """Calculate Intersection over Union (IoU)"""
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

    def run_and_eval(self):
        """Run evaluation pipeline and print results"""
        # New csv file and add header to result dataframe
        result_df = pd.DataFrame(columns=["IoU", "Similarity Metric", "Character Error Rate (CER)"])

        # Get prediction from detection model
        for image, ground_truth_bbox, ground_truth_text in zip(
            self.images, 
            self.ground_truth_detection, 
            self.ground_truth_ocr
        ):
            
            pred_bbox = self.model_detection.predict(image)

            # Get OCR result
            cropped_image = image[ground_truth_bbox[1]:ground_truth_bbox[3], ground_truth_bbox[0]:ground_truth_bbox[2]]
            pred_text = self.model_ocr.predict(cropped_image)

            # Calculate metrics
            iou = self.IoU(ground_truth_bbox, pred_bbox)
            similarity = self.similarity_metric(ground_truth_text, pred_text)
            cer = self.cer_metric(ground_truth_text, pred_text)

            # Add results to dataframe
            result_df = result_df.append({"IoU": iou, "Similarity Metric": similarity, "Character Error Rate (CER)": cer}, ignore_index=True)

        # Save results to csv
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
        model_detection = YoloDetection()
    
    if args.ocr_model == "yolo":
        model_ocr = YoloLicensePlateOCR()
    elif args.ocr_model == "easyocr":
        model_ocr = EasyOCR()

    # Initialize evaluator
    evaluator = Evaluate(
        model_detection=model_detection,
        model_ocr=model_ocr,
        images_path=args.images,
        ground_truth_detection_path=args.ground_truth_detection_path,
        ground_truth_ocr_path=args.ground_truth_ocr_path
    )

    # Run evaluation
    evaluator.run_and_eval()