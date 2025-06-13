from torchvision.models import resnet50
from torchvision import transforms
from filters.model_segmentation_filter import ModelSegmentationFilter
import torch
from PIL import Image

class FeatureVectorExtractor(ModelSegmentationFilter):
    def __init__(self, name: str, logger):
        super().__init__(name, logger=logger)
        self.model = self.load_model()

    def load_model(self):
        model = resnet50(pretrained=True)
        model.eval()
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
        return model

    def apply(self, data: dict):
        fingerprints = []
        for cutout in data["final_images"]:
        # Remove alpha channel for feature extraction
            if cutout.shape[2] == 4:
                rgb_cutout = cutout[..., :3]
            else:
                rgb_cutout = cutout
            img = Image.fromarray(rgb_cutout)

            input_tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                features = self.model(input_tensor)
            
            fingerprint = features.flatten()
            fingerprints.append(fingerprint.cpu().numpy())
            self.logger.info(f"Extracted fingerprint SIZE: {len(fingerprint)}")
        data["fingerprint"] = fingerprints
        self.logger.info(f"Extracted feature vectors: {fingerprints}")
        return data
    
    @property
    def transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])