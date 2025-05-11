from logging import Logger
from filters.filter import Filter
import torch
from torchvision import transforms as T
import numpy as np

class ModelSegmentationFilter(Filter):
    def __init__(self, name: str, model_file : str, logger: Logger):
        super().__init__(name, logger)
        if model_file is None:
            raise ValueError("Model file not set-up")
        
        self.name = name
        self.logger = logger
        self.resources_dir = f"resources/models/{model_file}"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def apply(self, data):
        if self.model is None:
            print("Model not loaded. Cannot apply filter.")
            return None
        if 'image' not in data:
            print("No image found in data.")
            return None
        
        image = data['image']
        try:
            if isinstance(image, np.ndarray):
                transform = T.ToTensor()
                image = transform(image)

            image = image.to(self.device)

            # Forward pass through the model
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))

            data['segmentation_data'] = output
            print(f"Model Prediction: {output}")
            return data
        
        except Exception as e:
            print(f"Error during model application: {e}")
            return None