import torch 
import torch.nn as nn
import cv2
from ultralytics import YOLO

class YoloReID():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.features_model = nn.Sequential(*list(self.model.model.children())[0][:-1])
    
    def image_feature(self, image):
        image = cv2.resize(image, (64,64))
        input_image = torch.Tensor(image)
        # Convert the 3D tensor to a 4D tensor with shape (1, 3, 64, 64)
        input_image = input_image.permute(2, 0, 1)  # Change the order of the dimensions to (3, 64, 64)
        input_image = input_image.unsqueeze(0)  # Add a new dimension at index 0
        features = self.features_model(input_image).flatten().cpu().detach().numpy()

        return features