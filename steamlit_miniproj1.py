import io
import os
import urllib.request

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Define model and preprocessing steps
model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@st.cache(allow_output_mutation=True)
def load_labels():
    # Download labels file if it doesn't exist
    if not os.path.exists('imagenet_classes.txt'):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt',
            'imagenet_classes.txt'
        )

    # Load labels into memory
    with open('imagenet_classes.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    return labels


def predict(image):
    # Preprocess image
    img_tensor = preprocess(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)

    # Get top 5 predicted class IDs and probabilities
    probs, class_ids = torch.topk(torch.softmax(output, dim=1), k=5)

    # Get class labels for the predicted class IDs
    labels = load_labels()

    # Display predictions
    st.write('Top 5 Predictions:')
    for i in range(5):
        st.write(f'{i+1}. {labels[class_ids[0][i]]}: {probs[0][i]*100:.2f}%')


def main():
    st.title('Image Classification Demo')

    # Prompt user to upload an image
    uploaded_image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])

    # Make prediction if an image has been uploaded
    if uploaded_image is not None:
        # Load image into memory
        image = Image.open(io.BytesIO(uploaded_image.read()))

        # Display image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        predict(image)


if __name__ == '__main__':
    main()
