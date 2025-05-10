import torch
import numpy as np
from filters.filter import Filter
from logging import Logger
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import torchvision
import torchvision.transforms as T

"""Example filter for neural network selection preprocessing."""

class ModelSelectionFilter(Filter):
    def __init__(self, name: str, logger: Logger):
        super().__init__(name, logger)
        model_paths = {"yolo" : "yolo.pt", "mask" : "maskrcnn.pth"}
        nn_model = model_paths[name]
        self.resources_dir = f"resources/models/{nn_model}"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.load_model()
        
    def get_model(self, num_classes):
        # Load pre-trained Mask R-CNN with ResNet-50 FPN backbone
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
        # Get the number of input features for the bounding box classifier
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained bounding box head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        if self.name == "mask":
            # Get the number of input features for the mask classifier
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256  # Standard hidden layer dimension for Mask R-CNN
            # Replace the pre-trained mask head with a new one
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        return model
    
    def load_model(self):
        try:
            # Define the default folder for model files
            
            # Construct the full path to the model file
            #model_path = os.path.join(resources_dir, model_filename)
            
            
            # # Check if the model file exists
            # if not os.path.isfile(model_path):
            #     raise FileNotFoundError(f"Model file '{model_filename}' not found in resources directory.")

            # Load the model
            model = self.get_model(num_classes=2)
            #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            model.load_state_dict(torch.load(self.resources_dir))
            model.to(self.device)
            model.eval()  # Set the model to evaluation mode
            #print(f"Successfully loaded model from {model_path}")
            return model
        
        except Exception as e:
            print(f"Error loading model from {self.resources_dir}: {e}")
            return None

    def apply(self, image):
        # Check if the model is loaded properly
        if self.model is None:
            print("Model not loaded. Cannot apply filter.")
            return None
        
        try:
            # Move the image and model to the appropriate device (CPU/GPU)
            if isinstance(image, np.ndarray):
                transform = T.ToTensor()
                image = transform(image)
                
            # self.model.to(self.device)
            image = image.to(self.device)

            # Forward pass through the model
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))  # Add batch dimension
              #  _, predicted = torch.max(output, 1)

            print(f"Model Prediction: {output}")
            return output
        
        except Exception as e:
            print(f"Error during model application: {e}")
            return None
