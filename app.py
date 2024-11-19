import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
from PIL import Image
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "./Dataset"
dataset = datasets.ImageFolder(root=data_dir)
classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_YellowLeaf__Curl_Virus'
]
print("Classes:", classes)

class EnhancedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base(x)
        return self.fc(x)

base_model = models.resnet18(weights=None)
enhanced_model = EnhancedModel(base_model, len(classes)).to(device)
enhanced_model.load_state_dict(torch.load("model.pth", map_location=device))
enhanced_model.eval()
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("Plant Disease Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = enhanced_model(input_tensor)
        _, predicted = torch.max(output, 1)

    st.write(f"Predicted Class: {classes[predicted.item()]}")
