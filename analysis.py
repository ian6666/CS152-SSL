import matplotlib.pyplot as plt
import os
import numpy as np
from model import confusion_matrix, get_model, plot_confusion_matrix, predict_with_images, accuracy
import torch
import custom_utils
from tqdm import tqdm
import pdb


def plot_losses(model_checkpoints, model_names, saving_dir, ymin = None, ymax = None):
    
    plt.figure(figsize=(10, 6))

    for model_checkpoint, model_name in zip(model_checkpoints, model_names):
        data = np.load(os.path.join("model_checkpoints", f'{model_checkpoint}', "mixmatch_model_loss.npy"))
        train_losses = data[0]
        train_accuracies = data[1]
        validation_losses = data[2]
        validation_accuracies = data[3]

        epochs = range(1, len(train_losses) + 1)

        # Plotting training losses
        plt.plot(epochs, train_losses, label=f'Training Loss - {model_name}')

        # Plotting validation losses
        plt.plot(epochs, validation_losses, '--', label=f'Validation Loss - {model_name}')

    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()


    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

    # Save plot
    plt.savefig(os.path.join(saving_dir, f"{model_checkpoints}_losses.pdf"))
    plt.savefig(os.path.join(saving_dir, f"{model_checkpoints}_losses.png"))
    plt.close()

def plot_accuracies(model_checkpoints, model_names, saving_dir, ymin = None, ymax = None):
    
    plt.figure(figsize=(10, 6))

    for model_checkpoint, model_name in zip(model_checkpoints, model_names):
        data = np.load(os.path.join("model_checkpoints", f'{model_checkpoint}', "mixmatch_model_loss.npy"))
        train_accuracies = data[1]
        validation_accuracies = data[3]

        epochs = range(1, len(train_accuracies) + 1)

        # Plotting training losses
        plt.plot(epochs, train_accuracies, label=f'Training Accuracies - {model_name}')

        # Plotting validation losses
        plt.plot(epochs, validation_accuracies, '--', label=f'Validation Accuracies - {model_name}')

    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.legend()
    plt.tight_layout()

    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)

    # Save plot
    plt.savefig(os.path.join(saving_dir, f"{model_checkpoints}_accuracies.pdf"))
    plt.savefig(os.path.join(saving_dir, f"{model_checkpoints}_accuracies.png"))
    plt.close()

# generate a confusion matrix for each model 
def load_model_confusion_matrix(model_checkpoints, saving_dir, device, num_classes=5):
    model, loss_fn, optimizer = get_model(num_classes, 0.001)
    for model_checkpoint in model_checkpoints:
        model.load_state_dict(torch.load(f'./model_checkpoints/{model_checkpoint}/best_model.pth'))

        model.eval()

        model = model.to(device)

        batch_size = 16
        labeled_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/train/_annotations.coco.json')
        labeled_dataloader = custom_utils.get_labeled_data("LabeledData/train", labeled_annotations, batch_size, False, return_filename=True)

        labeled_test_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/test/_annotations.coco.json')
        labeled_test_dataloader = custom_utils.get_labeled_data("LabeledData/test", labeled_test_annotations, batch_size, False, return_filename=True)

        current_cm = torch.zeros((num_classes, num_classes))
        for x,y,filename in tqdm(labeled_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            current_cm += confusion_matrix(x,y, model, num_classes, None)

        
        plot_confusion_matrix(current_cm, os.path.join(saving_dir, f"{model_checkpoint}_cm.pdf"))
        plot_confusion_matrix(current_cm, os.path.join(saving_dir, f"{model_checkpoint}_cm.png"))


def predict_and_save_images(model_checkpoints, saving_dir, device, num_classes=5):
    model, loss_fn, optimizer = get_model(num_classes, 0.001)
    for model_checkpoint in model_checkpoints:
        model.load_state_dict(torch.load(f'./model_checkpoints/{model_checkpoint}/best_model.pth'))

        model.eval()

        model = model.to(device)

        batch_size = 16
        labeled_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/train/_annotations.coco.json')
        labeled_dataloader = custom_utils.get_labeled_data("LabeledData/train", labeled_annotations, batch_size, True, return_filename=True)

        labeled_test_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/test/_annotations.coco.json')
        labeled_test_dataloader = custom_utils.get_labeled_data("LabeledData/test", labeled_test_annotations, batch_size, True, return_filename=True)

        current_cm = torch.zeros((num_classes, num_classes))
        for x,y,filename in tqdm(labeled_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            predict_with_images(x,y,model, "LabeledData/test", filename, f"./model_checkpoints/{model_checkpoint}/")
            break

def test_accuracies(model_checkpoint, device, num_classes=5):
    model, loss_fn, optimizer = get_model(num_classes, 0.001)
    for model_checkpoint in model_checkpoints:
        model.load_state_dict(torch.load(f'./model_checkpoints/{model_checkpoint}/best_model.pth'))

        model.eval()

        model = model.to(device)

        batch_size = 16
        labeled_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/train/_annotations.coco.json')
        labeled_dataloader = custom_utils.get_labeled_data("LabeledData/train", labeled_annotations, batch_size, True, return_filename=True)

        labeled_test_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/test/_annotations.coco.json')
        labeled_test_dataloader = custom_utils.get_labeled_data("LabeledData/test", labeled_test_annotations, batch_size, True, return_filename=True)

        all_batch_accuracies = []
        for x,y,filename in tqdm(labeled_test_dataloader):
            x = x.to(device)
            y = y.to(device)
            batch_accuracy = accuracy(x, y, model)
            all_batch_accuracies.append(batch_accuracy)
        all_batch_accuracies = torch.tensor(all_batch_accuracies)
        test_accuracies = torch.mean(all_batch_accuracies)
        print(f"Model {model_checkpoint} has accuracy {test_accuracies} on test data.")



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # varying learning rate
    model_checkpoints = [25, 26, 27, 28]
    model_names = ["lr=1e-3", "lr=1e-4", "lr=1e-5", "lr=1e-6"]
    saving_dir = "12_12_2023_results/"
    plot_losses(model_checkpoints, model_names, saving_dir, 0, 5)
    plot_accuracies(model_checkpoints, model_names, saving_dir, 0, 1)

    load_model_confusion_matrix(model_checkpoints, saving_dir, device, num_classes=5)
    predict_and_save_images(model_checkpoints, saving_dir, device, num_classes=5)
    test_accuracies(model_checkpoints, device, num_classes=5)

    # varying lambda_U
    model_checkpoints = [29, 30, 31, 32]
    model_names = ["lambda_U=200", "lambda_U=10", "lambda_U=1", "lambda_U=0"]
    saving_dir = "12_12_2023_results/"
    plot_losses(model_checkpoints, model_names, saving_dir, 0, 5)
    plot_accuracies(model_checkpoints, model_names, saving_dir, 0, 1)

    load_model_confusion_matrix(model_checkpoints, saving_dir, device, num_classes=5)
    predict_and_save_images(model_checkpoints, saving_dir, device, num_classes=5)
    test_accuracies(model_checkpoints, device, num_classes=5)

    # varying temperature
    model_checkpoints = [33, 34]
    model_names = ["temperature=0.3", "temperature=0.7"]
    saving_dir = "12_12_2023_results/"
    plot_losses(model_checkpoints, model_names, saving_dir, 0, 5)
    plot_accuracies(model_checkpoints, model_names, saving_dir, 0, 1)

    load_model_confusion_matrix(model_checkpoints, saving_dir, device, num_classes=5)
    predict_and_save_images(model_checkpoints, saving_dir, device, num_classes=5)
    test_accuracies(model_checkpoints, device, num_classes=5)

    # varying alpha
    model_checkpoints = [35, 36]
    model_names = ["alpha=0.3", "alpha=0.5"]
    saving_dir = "12_12_2023_results/"
    plot_losses(model_checkpoints, model_names, saving_dir, 0, 5)
    plot_accuracies(model_checkpoints, model_names, saving_dir, 0, 1)

    load_model_confusion_matrix(model_checkpoints, saving_dir, device, num_classes=5)
    predict_and_save_images(model_checkpoints, saving_dir, device, num_classes=5)
    test_accuracies(model_checkpoints, device, num_classes=5)

    # normal resnet16
    model_checkpoints = ['res']
    model_names = ["ResNet18"]
    saving_dir = "12_12_2023_results/"
    plot_losses(model_checkpoints, model_names, saving_dir, 0, 5)
    plot_accuracies(model_checkpoints, model_names, saving_dir, 0, 1)

    load_model_confusion_matrix(model_checkpoints, saving_dir, device, num_classes=5)
    predict_and_save_images(model_checkpoints, saving_dir, device, num_classes=5)
    test_accuracies(model_checkpoints, device, num_classes=5)
