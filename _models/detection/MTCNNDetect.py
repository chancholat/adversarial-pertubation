import cv2
import numpy as np
import torch
import torch.nn as nn
import os

from .base import BaseDetector
from ._models.MTCNN.MTCNN import create_mtcnn_net

class MTCNNDetector(BaseDetector):
    def __init__(self):
        super(MTCNNDetector, self).__init__()
        
        # Get the absolute path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate two levels up to reach the root directory
        root_dir = os.path.dirname(os.path.dirname(script_dir))
        # Construct the model path relative to the discovered root
        self.model_pnet_path = os.path.join(root_dir, 'assets', 'pretrained', 'License_Plate_Detection_Pytorch', 'MTCNN', 'pnet_Weights')
        self.model_onet_path = os.path.join(root_dir, 'assets', 'pretrained', 'License_Plate_Detection_Pytorch', 'MTCNN', 'onet_Weights')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.compute_loss = ComputeLoss(self.model.model.model)

    def preprocess(self, images):
        pass
        # images = np.stack(images, axis=0)
        # return images

    def postprocess(self, adv_images):
        adv_images = adv_images.detach().cpu().numpy().transpose(1,2,0) * 255.0
        adv_images = cv2.cvtColor(adv_images, cv2.COLOR_RGB2BGR)
        return adv_images

    def forward(self, adv_images, targets):
        pass

    def detect(self, images):
        predictions = []

        # if len(images.shape) == 3:
        #     images = images.unsqueeze(0)

        for img in images:
            bboxes = create_mtcnn_net(img, (50, 15), self.device, p_model_path=self.model_pnet_path, o_model_path=self.model_onet_path)
            predictions.append(bboxes)

        return predictions

    def make_targets(self, predictions, images):
        pass
        # targets = []
        # for i, (pred, image) in  enumerate(zip(predictions, images)):
        #     h, w, _ = image.shape

        #     # extract obj confidence instead of class number (?), xmin, ymin, xmax, ymax
        #     pred = np.array([[item[4], item[0], item[1], item[2], item[3]] for item in pred])

        #     if len(pred) == 0:
        #         pred = np.zeros((0, 6))

        #     nl = len(pred)
        #     target = torch.zeros((nl, 6))
        #     # convert xyxy to xc, yc, wh
        #     pred[:, 1:5] = xyxy2xywhn(pred[:, 1:5], w=w, h=h, clip=True, eps=1e-3)
        #     target[:, 1:] = torch.from_numpy(pred)

        #     # add image index for build target
        #     target[:, 0] = i
        #     targets.append(target)

        # return torch.cat(targets)

    def get_bboxes(self, predictions):
        bboxes = []
        
        for pred in predictions:
            # pred can contain  multiple bboxes
            bbox_list = [[int(point) for point in box[0:4]] for box in pred]
            bboxes.append(bbox_list)
        
        return bboxes
    

# mtcnn_det = MTCNNDetector()