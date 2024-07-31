import os,pickle
import pandas as pd
import config
from utils import save_obj

import gcsfs
from google.cloud import storage

#provide service account json to get data from bucket
credential_path = 'storage_service_client.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
#initialise storange client 
storage_client = storage.Client()
#Load integer to character mapping dictionary from GCS bucket
gcs_file_system = gcsfs.GCSFileSystem(project="Project_name")

if __name__ == "__main__":

    image_path = config.image_path
    label_path = config.label_path
    char2int_path = config.char2int_path
    int2char_path = config.int2char_path
    data_file_path = config.data_file_path

    # Read the labels
    labels = pd.read_table(label_path, header=None)
    # Fill missing values with "null"
    labels.fillna("null", inplace=True)

    # Get all image IDs
    image_files = os.listdir(image_path)
    image_files.sort()
    # Create full paths for the images
    image_files = [os.path.join(image_path, i) for i in image_files]

    # Find the unique characters in the labels
    unique_chars = list({l for word in labels[0] for l in word})
    unique_chars.sort()
    # Create maps from character to integer and integer to character
    char2int = {a: i+1 for i, a in enumerate(unique_chars)}
    int2char = {i+1: a for i, a in enumerate(unique_chars)}

    # Save the maps
    save_obj(char2int, char2int_path)
    print(f'saved char2int to {char2int_path}')
    save_obj(int2char, int2char_path)
    print(f'saved int2char to {int2char_path}')

    # Create data file containing the image paths and the labels
    data_file = pd.DataFrame({"images": image_files, "labels": labels[0]})
    data_file.to_csv(data_file_path, index=False)
    print(f'saved data_file to {data_file_path}')
    gcs_int2char = "gs://deep_learning_project_for_text_detection_in_images/data/int2char.pkl"
    with gcs_file_system.open(gcs_int2char,'wb') as f:
        int2char = pickle.dump(int2char, f)
    print(f'Sucessfully saved int2char file to {gcs_int2char}')
    gcs_char2int= "gs://deep_learning_project_for_text_detection_in_images/data/char2int.pkl"
    with gcs_file_system.open(gcs_char2int,'wb') as f:
        char2int = pickle.dump(char2int, f)
    print(f'Sucessfully saved char2int file to {gcs_char2int}')
    data_file.to_csv("gs://deep_learning_project_for_text_detection_in_images/data/data_file.csv", index=False)
    print(f'Sucessfully saved data_file to the bucket')

    
