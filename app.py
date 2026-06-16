import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageOps

from model import DigitCNN

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess Image
    image = ImageOps.invert(image.convert("L"))
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Predicted Digit: {predicted.item()}")