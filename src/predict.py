import cv2
import torch
import albumentations
from utils import load_obj
from source.network import ConvRNN
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--test_img", default=" default_path", help="path to test image")
    parser.add_argument("--model_path", default="output/models/model.pth", help="path to the saved model")
    parser.add_argument("--int2char_path", default=" default_path", help="path to int2char")
    opt = parser.parse_args()

    # Load integer to character mapping dictionary
    int2char = load_obj(opt.int2char_path)
    # Number of classes
    n_classes = len(int2char)

    # Create model object
    model = ConvRNN(n_classes)
    # Load model weights
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'))
    print('Sucessfully loaded the model')
    # Port model to cuda if available
    if torch.cuda.is_available():
        model.cuda()
    # Set model mode to evaluation
    model.eval()

    # Check if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Load and pre-process the image
    img = cv2.imread(opt.test_img)
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
    print(res)
