import torchvision
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms,models,datasets
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim
device = "cuda" if torch.cuda.is_available() else "cpu"
import glob
import matplotlib.pyplot as plt
from glob import glob
import torchvision.transforms as transforms
from custom_utils import get_annotations
from custom_utils import get_labeled_data
import pdb
import os

classes = {0:"Crimp",
           1:"Jug",
           2:"Pinch",
           3:"Pocket",
           4:"Sloper"}

def get_model(num_classes, lr):
    model = models.resnet18(pretrained=True)
    
    # Freeze the parameters of the model
    for param in model.parameters():
        param.requires_grad = False
    
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr)

    return model.to(device), loss_fn, optimizer

def train_batch(x, y, model, optimizer, loss_fn):
    x = x.to(device)
    y = y.to(device)
    model.train()  # Set the model to training mode
    prediction = model(x)
    # pdb.set_trace()
    # For multi-class classification, ensure y is in the correct shape
    # and of a suitable dtype, like long. 
    # This might require modification depending on how your labels 'y' are provided.

    batch_loss = loss_fn(prediction, y)  # CrossEntropyLoss is used for multi-class
    
    optimizer.zero_grad()  # Zero the gradients before backward pass
    batch_loss.backward()  # Compute the gradients
    optimizer.step()  # Update parameters
    
    return batch_loss.item()

def accuracy(x, y, model):
    x = x.to(device)
    y = y.to(device)
    if len(y.shape) == 2:
        y = torch.argmax(y, dim=1)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(x)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct_count = (predicted_classes == y).sum().item()
        total_count = y.size(0)
        return correct_count / total_count
    
def predict(x, model):
    predictions = model(x)
    predicted_classes = torch.argmax(predictions, dim=1)
    return predicted_classes, [classes[predicted_class] for predicted_class in predicted_classes]

def predict_with_images(x, y, model, saving_dir):
    predicted_classes_num, class_names = predict(x, model)
    
    for input in range(len(x)):
        plt.imshow(input.permute([1,2,0]).detach().cpu())
        plt.set_title(f'True: {classes[y[input].item()]}, Pred: {class_names[input]}')
        plt.tight_layout()
        filename = f'{input:05d}.png'
        plt.savefig(os.path.join(saving_dir, filename), bbox_inches='tight')


def predict_proba(x, model):
    predictions = model(x)
    return torch.nn.functional.softmax(predictions)

if __name__=="__main__":
    labeled_json_path = "ClimbingHoldDetection-15/train/_annotations.coco.json"
    annotations = get_annotations(labeled_json_path)
    trn_dl = get_labeled_data("LabeledData", annotations)
    model, loss_fn, optimizer = get_model(5)

    train_losses, train_accuracies = [], []
    n_epoch = 1

    print("All losses and accuracies are for each epoch")
    for epoch in range(n_epoch):
        train_epoch_losses, train_epoch_accuracies = [], []

        for ix, batch in enumerate(iter(trn_dl)):
            try:
                x, y = batch
                x, y = x.to(device), y.to(device)
                x = x.float()  # Convert to float
                y = y.long()   # Convert to long for classification labels

                batch_loss = train_batch(x, y, model, optimizer, loss_fn)
                train_epoch_losses.append(batch_loss)
            except Exception as e:
                print(f"An error occurred at batch {ix}: {e}")
                continue  # Skip this batch

        train_epoch_loss = np.array(train_epoch_losses).mean()

        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = x.float()  # Convert to float
            y = y.long()   # Convert to long for classification labels
            

            batch_accuracy = accuracy(x, y, model)
            train_epoch_accuracies.append(batch_accuracy)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        print(f"Epoch {epoch + 1}/{n_epoch}, Training Loss: {train_epoch_loss:.4f}, Training Accuracy: {train_epoch_accuracy:.4f}")
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)

    # epochs = np.arange(5)+1
    # import matplotlib.ticker as mtick
    # import matplotlib.pyplot as plt
    # import matplotlib.ticker as mticker

    # plt.plot(np.arange(len(train_accuracies)), train_accuracies, 'b', label='Training accuracy')
    # # plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    # plt.title('Training and validation accuracy with ResNet18 \nand 1K training data points')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]) 
    # plt.legend()
    # plt.grid('off')
    # plt.show()