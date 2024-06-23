import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Define transformations for input images
input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the VGG16 model
vgg16_model = torch.load('vgg16_model.pth', map_location=torch.device('cpu'))
vgg16_model.eval()

# Load the ResNet model
resnet_model = torch.load('resnet18_model.pth', map_location=torch.device('cpu'))
resnet_model.eval()

# Function to predict with VGG16 model
def predict_vgg16(img):
    image = Image.open(img)
    image = input_transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = torch.sigmoid(vgg16_model(image))
        prediction = torch.round(output).squeeze().item()  # Round to 0 or 1
    
    return "Painting" if prediction == 0 else "Photo"

# Function to predict with ResNet model
def predict_resnet(img):
    image = Image.open(img)
    image = input_transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = torch.sigmoid(resnet_model(image))
        prediction = torch.round(output).squeeze().item()  # Round to 0 or 1
    
    return "Painting" if prediction == 0 else "Photo"

# Define Gradio interfaces
iface_vgg16 = gr.Interface(
    fn=predict_vgg16,
    inputs=gr.Image(type='filepath'),
    outputs=gr.Textbox(),
    title="VGG16 Image Classifier",
    description="Upload an image and classify it as 'Painting' or 'Photo' using VGG16 model."
)

iface_resnet = gr.Interface(
    fn=predict_resnet,
    inputs=gr.Image(type='filepath'),
    outputs=gr.Textbox(),
    title="ResNet Image Classifier",
    description="Upload an image and classify it as 'Painting' or 'Photo' using ResNet model."
)

iface_combined = gr.Interface(
    fn=lambda img: "\n".join([
        "VGG16 Prediction: " + predict_vgg16(img),
        "ResNet Prediction: " + predict_resnet(img)
    ]),
    inputs=gr.Image(type='filepath'),
    outputs=gr.Textbox(),
    title="Combined Image Classifier",
    description="Upload an image and get predictions from both VGG16 and ResNet models."
)

# Launch the Gradio interfaces

iface_combined.launch(share=True)
#iface_vgg16.launch(share=True)
#iface_resnet.launch(share=True)