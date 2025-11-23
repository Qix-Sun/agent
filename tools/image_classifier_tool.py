# File: ./tools/image_classifier_tool.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ImageClassifierTool:
    def __init__(self, model_path="./tools/models/resnet_cataract_classifier.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.class_map = {0: "B超图像", 1: "眼底图图像"}

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        model = models.resnet18(weights=None)

        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(checkpoint['class_to_idx']))
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()  # 设置为评估模式
        return model

    def classify_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted = torch.argmax(output, dim=1).item()
            label = self.class_map[predicted]
        return {"type": label}

# Example usage
if __name__ == "__main__":
    tool = ImageClassifierTool()
    image_file = ""
    prediction = tool.classify_image(image_file)
    print(f"Image Classification Result: {prediction}")