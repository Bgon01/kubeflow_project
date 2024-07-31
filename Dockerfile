FROM gcr.io/imaya-2/deep-learning-project/base_image:latest
RUN apt upgrade && apt update && apt install -y \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    python3-pip \
    python3-dev  \
    git
COPY . deep_learning_folder/
RUN pip3 install --upgrade pip
RUN pip3 install  --no-cache -r deep_learning_folder/requirements.txt
ENTRYPOINT [ "sh", "-c" ]
CMD ["cd deep_learning_folder && python3 flask_predict.py"]

#docker build -t img_txt_fast:latest . 
#docker run -p 9999:9999 img_txt_fast:latest
# gcloud builds submit --timeout=1800s --tag gcr.io/imaya-2/deep-learning-project/base_image:latest .

#created cloud trigger and pushing to master