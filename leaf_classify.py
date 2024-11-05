from PIL import Image
import torchvision.transforms as TF
import sys
sys.path.append('.')
import CNN
import torch
import numpy as np


new_img = Image.open('leaf_images/preprocessed_image.jpg').convert('RGB').resize((224, 224))
img = TF.ToTensor()(new_img)
img = img.unsqueeze(0)
img = img / 255.0

# Save the preprocessed image
# img_numpy = img.squeeze(0).permute(1, 2, 0).numpy()
# img = Image.fromarray((img_numpy * 255).astype('uint8'))

# Load the model and make a prediction
model = CNN.CNN(39)
model.load_state_dict(torch.load('plant_disease_model_1_latest.pt', map_location=torch.device('cpu')))
prediction = model(img).detach().numpy()

predicted_class_idx = np.argmax(prediction[0])

class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Get the predicted label
predicted_label = class_labels[predicted_class_idx]

print(f"Predicted Label: {predicted_label}")

# Convert the prediction to a JSON-serializable format
# prediction_str = str(prediction[0][0])
# if prediction[0][0] < 0:
#     pred = "Not Diseased"
# else:
#     pred = "Diseased"

# print(pred)

