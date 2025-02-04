import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# Load the trained model
def load_model():
    # Define the model architecture (must match the one used during training)
    model = models.resnet18(pretrained=False)
    num_classes = 44  # Update this to match the number of rune letters in your dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load("model/rune_classifier.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

# Make a prediction
def predict(image, model, class_names):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

# Streamlit app
def main():
    st.title("Rune Alphabet Prediction")
    st.write("Upload an image of a rune, and the model will predict the corresponding letter.")

    # Load the model
    model = load_model()

    # Define class names (update this to match your dataset)
    class_names = ['algiz',
 'algiz_inverted',
 'ansur',
 'ansur_inverted',
 'berkana',
 'berkana_inverted',
 'dagaz',
 'eivaz',
 'evaz',
 'evaz_inverted',
 'fehu',
 'fehu_inverted',
 'gebo',
 'hagalaz',
 'inguz',
 'isa',
 'kennaz',
 'kennaz_inverted',
 'laguz',
 'laguz_inverted',
 'mannaz',
 'mannaz_inverted',
 'mimir',
 'mimir_inverted',
 'nautiz',
 'odal',
 'odal_inverted',
 'pert',
 'pert_inverted',
 'raido',
 'raido_inverted',
 'sovelo',
 'star',
 'teivaz',
 'teivaz_inverted',
 'turisaz',
 'turisaz_inverted',
 'uruz',
 'uruz_inverted',
 'vunio',
 'vunio_inverted',
 'yera',
 'ziu',
 'ziu_inverted']

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Make a prediction
        predicted_class = predict(image_tensor, model, class_names)

        # Display the result
        st.success(f"Predicted Rune: **{predicted_class}**")

# Run the app
if __name__ == "__main__":
    main()