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
import custom_utils 
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
    batch_size = 16
    labeled_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/train/_annotations.coco.json')
    labeled_dataloader = custom_utils.get_labeled_data("LabeledData/train", labeled_annotations, batch_size, False)

    labeled_valid_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/valid/_annotations.coco.json')
    labeled_valid_dataloader = custom_utils.get_labeled_data("LabeledData/valid", labeled_valid_annotations, batch_size, False)
    model, loss_fn, optimizer = get_model(5,lr=1e-4)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    n_epoch = 50

    print("All losses and accuracies are for each epoch")

    max_accuracy = 0
    savingPath = os.path.join("model_checkpoints", "res")
    if not os.path.exists(savingPath):
        os.makedirs(savingPath)
    for epoch in range(n_epoch):
        train_epoch_losses, train_epoch_accuracies = [], []
        labeled_trn_dl_iter = iter(labeled_dataloader)
        for ix, batch in enumerate(labeled_trn_dl_iter):
            try:
                x, y = batch
                x, y = x.to(device), y.to(device)
                x = x.float()  # Convert to float
                y = y.long()   # Convert to long for classification labels

                batch_loss = train_batch(x, y, model, optimizer, loss_fn)
                train_epoch_losses.append(batch_loss)
                batch_accuracy = accuracy(x, y, model)
                train_epoch_accuracies.append(batch_accuracy)

            except Exception as e:
                print(f"An error occurred at batch {ix}: {e}")
                continue  # Skip this batch

        train_epoch_loss = np.array(train_epoch_losses).mean()
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        

        model.eval()
        labeled_val_dl_iter = iter(labeled_valid_dataloader)
        val_epoch_losses, val_epoch_accuracies = [], []
        for ix, batch in enumerate(labeled_val_dl_iter):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = x.float()  # Convert to float
            y = y.long()   # Convert to long for classification labels

            batch_loss = loss_fn(model(x),y)
            val_epoch_losses.append(batch_loss.item())
            batch_accuracy = accuracy(x, y, model)
            val_epoch_accuracies.append(batch_accuracy)
            
        val_epoch_loss = np.mean(val_epoch_losses)
        val_epoch_acc = np.mean(val_epoch_accuracies)
        if val_epoch_acc>max_accuracy:
            max_accuracy = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(savingPath, "best_model.pth"))

        print(f"Epoch {epoch + 1}/{n_epoch}, Training Loss: {train_epoch_loss:.4f}, Training Accuracy: {train_epoch_accuracy:.4f}, Validation Loss:{val_epoch_loss:.4f}, Validation Accuracy:{val_epoch_acc:4f}")
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        result = np.vstack([np.array(train_losses), np.array(train_accuracies), np.array(val_losses), np.array(val_accuracies)])
        np.save(os.path.join(savingPath, "mixmatch_model_loss.npy"), result)

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