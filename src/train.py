import torch
import config
import numpy as np
import pandas as pd
import torch.nn as nn
import pickle
from tqdm import tqdm
from utils import load_obj, save_obj
from source.network import ConvRNN
from source.dataset import TRSynthDataset
from sklearn.model_selection import train_test_split


credential_path = 'storage_service_client.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
#initialise storange client 
storage_client = storage.Client()
#Load integer to character mapping dictionary from GCS bucket
gcs_file_system = gcsfs.GCSFileSystem(project="imaya-2")
gcs_int2char = "gs://deep_learning_project_for_text_detection_in_images/data/int2char.pkl"
with gcs_file_system.open(gcs_int2char,'rb') as f:
    int2char = pickle.load(f)
print('Sucessfully loaded int2char file')
gcs_char2int= "gs://deep_learning_project_for_text_detection_in_images/data/char2int.pkl"
with gcs_file_system.open(gcs_char2int,'rb') as f:
    char2int = pickle.load(f)
print('Sucessfully loaded char2int file')
gcs_data_file = "gs://deep_learning_project_for_text_detection_in_images/data/data_file.csv"
data_file = pd.read_csv(gcs_data_file)
print('Sucessfully loaded data_file from bucket')

def train(model, dataloader, criterion, device, optimizer=None, test=False):
    """
    Function to train the model
    :param network: Model object
    :param loader: data loader
    :param loss_fn: loss function
    :param dvc: device (cpu or cuda)
    :param opt: optimizer
    :param test: True for validation (gradients won't be updated)
    :return: Average loss for the epoch
    """

    # Set mode to train or validation
    if test:
        model.eval()
    else:
        model.train()
    loss = []
    for inp, tgt, tgt_len in tqdm(dataloader):
        # Move tensors to device
        inp = inp.to(device)
        tgt = tgt.to(device)
        tgt_len = tgt_len.to(device)
        # Forward pass
        out = model(inp)
        out = out.permute(1,0,2)
        # Calculate input lengths for the data points
        # All have equal length of 40 since all images in
        # our dataset are of equal length
        inp_len = torch.LongTensor([40] * out.shape[1])
        # Calculate CTC Loss
        log_probs = nn.functional.log_softmax(out, 2)
        loss_ = criterion(log_probs, tgt, inp_len, tgt_len)

        if not test:
            # Update weights during training
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

        loss.append(loss_.item())

    return np.mean(loss)


if __name__ == "__main__":

    data_file_path = config.data_file_path
    char2int_path = config.char2int_path
    int2char_path = config.int2char_path
    epochs = config.epochs
    batch_size = config.batch_size
    model_path = config.model_path

    # Read the data file

    data_file.fillna("null", inplace=True)

    # Load character to integer mapping dictionary
    #char2int = load_obj(char2int_path)

    #not using in this file but saving it to add into git
    #int2char = load_obj(int2char_path)
    # Check if cuda is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Split the data into train and validation sets
    train_file, valid_file = train_test_split(data_file, test_size=0.2)

    # Create train and validation datasets
    train_dataset = TRSynthDataset(train_file, char2int)
    valid_dataset = TRSynthDataset(valid_file, char2int)

    # Define loss function
    criterion = nn.CTCLoss(reduction="sum")
    criterion.to(device)

    # Number of classes
    n_classes = len(char2int)
    # Create model object
    model = ConvRNN(n_classes)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0005)

    # Define train and validation dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)

    # Training loop
    best_loss = 1e7
    for i in range(epochs):
        print(f"Epoch {i+1} of {epochs}...")
        # Run train function
        train_loss = train(model, train_loader, criterion, device, optimizer, test=False)
        # Run validation function
        valid_loss = train(model, valid_loader, criterion, device, test=True)
        print(f"Train Loss: {round(train_loss,4)}, Valid Loss: {round(valid_loss,4)}")
        if valid_loss < best_loss:
            print("Validation Loss improved, saving Model File...")
            # Save model object
            torch.save(model.state_dict(), model_path)
            best_loss = valid_loss

    # Save the maps
    save_obj(char2int, char2int_path)
    print(f'saved char2int to {char2int_path}')
    save_obj(int2char, int2char_path)
    print(f'saved int2char to {int2char_path}')

    # Create data file containing the image paths and the labels
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

