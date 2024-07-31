import json,os, pickle5 as pickle
import cv2 , numpy as np
import torch
import albumentations
from src.utils import load_obj
from src.source.network import ConvRNN
from argparse import ArgumentParser
from flask import Flask , request
# import gcsfs
# from google.cloud import storage

#initialise Flask app
app = Flask(__name__)
#provide service account json to get data from bucket
# credential_path = 'storage_service_client.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
#initialise storange client 
#storage_client = storage.Client()
# Load integer to character mapping dictionary from GCS bucket
# gcs_file_system = gcsfs.GCSFileSystem(project="imaya-2")
# gcs_json_path = "gs://deep_learning_project_for_text_detection_in_images/data/int2char.pkl"
# with gcs_file_system.open(gcs_json_path,'rb') as f:
#     int2char = pickle.load(f)
# print('Sucessfully loaded pickle file')

#int2char path
int2charPath = 'input/data/int2char.pkl'
int2char = load_obj(int2charPath)

model_path = 'output/models/model.pth'
# Number of classes
n_classes = len(int2char)
# Create model object
model = ConvRNN(n_classes)
# Load model weights
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# Port model to cuda if available
if torch.cuda.is_available():
    model.cuda()
# Set model mode to evaluation
model.eval()
# Check if cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Sucessfully loaded the model in {device}')

@app.route("/", methods=['GET'])
def greet():
    return 'Yayyyy! working'


@app.route("/predictTextFromImage", methods=['POST'])
def predict():
    #get image as a file in form data
    image_to_predict = request.files.get('img')
    scode = 200
    if image_to_predict is None:
        scode = 400
        return {'status_code': scode, 'response': [],'message':'Upload an Image to predict' }
    # convert the file from base64 string to numpy array to pass it to CV model
    #image = np.asarray(bytearray(image_to_predict.read()))
    image = np.asarray(bytearray(image_to_predict.read()))
   

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # pre-process the image
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img_aug = albumentations.Compose(
            [albumentations.Normalize(mean, std,
                                      max_pixel_value=255.0,
                                      always_apply=True)]
        )
    augmented = img_aug(image=img)
    img = augmented["image"]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    # Create batch dimension (batch of single image)
    img = torch.unsqueeze(img, 0)
    # Move the image array to CUDA if available
    img = img.to(device)

    # Take model output
    out = model(img)
    # Remove the batch dimension
    out = torch.squeeze(out, 0)
    # Take softmax over the predictions
    out = out.softmax(1)
    # Take argmax to make predictions for the
    # 40 timeframes
    pred = torch.argmax(out, 1)
    # Convert prediction tensor to list
    pred = pred.tolist()
    # Use 'ph' for the special character
    int2char[0] = "ph"
    # Convert integer predictions to character
    out = [int2char[i] for i in pred]

    # Collapse the output
    res = list()
    res.append(out[0])
    for i in range(1, len(out)):
        if out[i] != out[i - 1]:
            res.append(out[i])
    res = [i for i in res if i != "ph"]
    res = "".join(res)
    return {'status_code': scode, 'response': res,'message':'Success' }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 5000,debug=True)