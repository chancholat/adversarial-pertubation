from collections import Counter
from Levenshtein import distance
import numpy as np

class Evaluate:
    def __init__(self, model_detection, model_ocr):
        self.model_detection = model_detection  # Detection model
        self.model_ocr = model_ocr  # OCR model

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

    def forward(self, image, ground_truth_bbox, ground_truth_text):
        """Run evaluation pipeline and print results"""
        # Get prediction from detection model
        pred_bbox = self.model_detection.predict(image)

        # Get OCR result
        cropped_image = image[ground_truth_bbox[1]:ground_truth_bbox[3], ground_truth_bbox[0]:ground_truth_bbox[2]]
        pred_text = self.model_ocr.predict(cropped_image)

        # Calculate metrics
        iou = self.IoU(ground_truth_bbox, pred_bbox)
        similarity = self.similarity_metric(ground_truth_text, pred_text)
        cer = self.cer_metric(ground_truth_text, pred_text)

        # Print results
        print("Evaluation Results:")
        print(f"IoU: {iou:.4f}")
        print(f"Similarity Metric: {similarity:.4f}")
        print(f"Character Error Rate (CER): {cer:.4f}")

# Example usage (with placeholder models)
class DummyDetectionModel:
    def predict(self, image):
        # Placeholder: Replace with actual model logic
        return [50, 150, 50, 150]  # Example bbox [xmin, xmax, ymin, ymax]

class DummyOCRModel:
    def predict(self, image):
        # Placeholder: Replace with actual OCR logic
        return "SampleText"

if __name__ == "__main__":
    detection_model = DummyDetectionModel()
    ocr_model = DummyOCRModel()

    evaluator = Evaluate(detection_model, ocr_model)

    # Example input
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)  # Dummy image
    ground_truth_bbox = [40, 160, 40, 160]  # Ground truth bbox
    ground_truth_text = "SampleText"

    evaluator.forward(test_image, ground_truth_bbox, ground_truth_text)