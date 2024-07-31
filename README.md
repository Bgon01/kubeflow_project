# Text Detection based on Convolutional Recurrent Neural Network (CRNN)

This repository contains the code for the CRNN implementation on the [TRSynth100K Dataset](https://www.kaggle.com/eabdul/textimageocr) from [Kaggle](https://www.kaggle.com/).

### Installation

To install the dependencies run:
```
pip install -r requirements.txt
```

### TRSynth100K Dataset
The dataset contains 100k images containing texts in them. Each image is of size 40x160 pixels. The text in the images are given as labels. The goal is to identify the text in the image.

Download the data, unzip the file and place the folder ```TRSynth100K``` inside the ```data``` folder.

### Data processing
To process the data, run:
```
python preprocessing.py
```

### Train the model
To train the model, run:
```
python train.py
```

### Use the model to make predictions
To make predictions on an image, run:
```python predict.py ----test_img path_to_test_img```