from logging import Logger
from filters.filter import Filter
import torch
from torchvision import transforms as T
import numpy as np

class ModelSegmentationFilter(Filter):
    def __init__(self, name: str, model_file : str, logger: Logger, resources_path: str):
        super().__init__(name, logger)
        if model_file is None:
            raise ValueError("Model file not set-up")
        
        self.name = name
        self.logger = logger
        self.resources_dir = f"{resources_path}/models/{model_file}"
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
            def get_mask_features_hook(module, input, output):
                # 'output' here is the tensor of features from mask_fcn4
                data['mask_features'] = output
            
            self.logger.info(f"Applying {self.name} model segmentation filter.")
            hook = self.model.roi_heads.mask_predictor.conv5_mask.register_forward_hook(get_mask_features_hook)
            image_tensor = None
            if isinstance(image, np.ndarray):
                transform = T.ToTensor()
                image_tensor = transform(image)

            image_tensor = image_tensor.to(self.device)

            # Forward pass through the model
            with torch.no_grad():
                output = self.model(image_tensor.unsqueeze(0))
            hook.remove()  # Remove the hook after use

            data['segmentation_data'] = output.cpu().numpy() if isinstance(output, torch.Tensor) else output
            torch.cuda.empty_cache()  # Clear GPU memory
            return data
        
        except Exception as e:
            print(f"Error during model application: {e}")
            return None