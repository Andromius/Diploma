import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import random
# Define transformations
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)  # Convert PIL image to tensor
        return image, target
    

# Dataset class
def get_coco_dataset(img_dir, ann_file):
    return torchvision.datasets.CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )


# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train():
    # Initialize the model
    num_classes = 2 # Background + chair, human, table
    model = get_model(num_classes)

    # Load datasets
    train_dataset = get_coco_dataset(
        img_dir="images",
        ann_file="instances_default.json"
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(torch.cuda.is_available())

    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train_one_epoch(model, optimizer, data_loader, device, epoch):
        model.train()
        for images, targets in data_loader:
            # Move images to the device
            images = [img.to(device) for img in images]

            # Validate and process targets
            processed_targets = []
            valid_images = []
            for i, target in enumerate(targets):
                boxes = []
                labels = []
                for obj in target:
                    # Extract bbox
                    bbox = obj["bbox"]  # Format: [x, y, width, height]
                    x, y, w, h = bbox

                    # Ensure the width and height are positive
                    if w > 0 and h > 0:
                        boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
                        labels.append(obj["category_id"])

                # Only process if there are valid boxes
                if boxes:
                    processed_target = {
                        "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                        "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                    }
                    processed_targets.append(processed_target)
                    valid_images.append(images[i])  # Add only valid images

            # Skip iteration if no valid targets
            if not processed_targets:
                continue

            # Ensure images and targets are aligned
            images = valid_images

            # Forward pass
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")


    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        # Save the model's state dictionary after every epoch
        model_path = f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

def predict():
    num_classes = 2  # Background + chair + person + table

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load("fasterrcnn_resnet50_epoch_8.pth"))
    model.to(device)
    model.eval()  # Set the model to evaluation mode


    def prepare_image(image_path):
        image = Image.open(image_path).convert("RGB")  # Open image
        image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
        return image_tensor.to(device)



    # Load the unseen image


    # `prediction` contains:
    # - boxes: predicted bounding boxes
    # - labels: predicted class labels
    # - scores: predicted scores for each box (confidence level)
    COCO_CLASSES = {0: "Background", 1: "Graffiti"}

    def get_class_name(class_id):
        return COCO_CLASSES.get(class_id, "Unknown")
        
    # Draw bounding boxes with the correct class names and increase image size
    def draw_boxes(image, prediction, fig_size=(10, 10)):
        boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
        labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
        scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores
        print(prediction[0].keys())
        
        # Set a threshold for showing boxes (e.g., score > 0.5)
        threshold = 0.2
        
        # Set up the figure size to control the image size
        fig = plt.figure(figsize=fig_size)  # Adjust the figure size here

        arr = list(zip(boxes, labels, scores))

        for i, (box, label, score) in enumerate(arr):
            if score > threshold:
                x_min, y_min, x_max, y_max = box
                class_name = get_class_name(label)  # Get the class name
                plt.imshow(image)  # Display the image
                plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                                linewidth=2, edgecolor='r', facecolor='none'))
                plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')
        def on_key(event):
            if event.key == 'escape':
                plt.close()

        plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        plt.axis('off')  # Turn off axis
        plt.show()

        


    image_paths = ["images/06.jpg"]  
    # Display the image with bounding boxes and correct labels
    for path in image_paths:
        image_tensor = prepare_image(path)

        with torch.no_grad():  # Disable gradient computation for inference
            prediction = model(image_tensor)
        draw_boxes(Image.open(path), prediction, fig_size=(12, 10))  # Example of increased size

if __name__  == '__main__':
    #train()
    predict()