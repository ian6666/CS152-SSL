import mixmatch
from model import get_model
from model import accuracy
from model import predict
import custom_utils
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.nn.functional import one_hot
import torch.nn as nn
import pdb
import os
import time
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "--temperature",
    default=0.5
)

parser.add_argument(
    "--k_augment",
    default=2
)

parser.add_argument(
    "--alpha",
    default=0.75
)

parser.add_argument(
    "--lambda_U",
    default=100
)

parser.add_argument(
    "--lr",
    default=1e-3
)

parser.add_argument(
    "--trial",
    required = True
)


def mixmatch_train_step(model, images_X, labels_X, images_U, loss_fn, optimizer, batch_size, num_classes, K_transforms, device, alpha, T, K, lambda_U):
    model.train()
    X_prime, p_X, U_prime, p_U = mixmatch.mixMatch(model, images_X, labels_X, images_U, batch_size, num_classes, K_transforms, device, alpha, T, K)
    optimizer.zero_grad()
    loss = loss_fn(X_prime, U_prime, p_X, p_U, model, num_classes, lambda_U)
    loss.backward()
    optimizer.step()
    return loss.item()



def mixmatch_train(model, loss_fn, val_loss_fn, optimizer, n_epoch, labeled_trn_dl, unlabeled_trn_dl, labeled_val_dl, device, batch_size, num_classes, K_transforms, savingPath,  alpha, T, K, lambda_U, trial):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    max_val_accuracy = 0
    
    print("All losses and accuracies are for each epoch")
    for epoch in range(n_epoch):
        train_epoch_losses, train_epoch_accuracies = [], []
        # batch_loss = mixmatch_train_step(model, loss_fn, optimizer, batch_size, num_classes, K_transforms) #?
        # train_epoch_losses.append(batch_loss)
        labeled_trn_dl_iter = iter(labeled_trn_dl)
        unlabeled_trn_dl_iter = iter(unlabeled_trn_dl)
        
        for ix in range(len(labeled_trn_dl)):
            images_U = next(unlabeled_trn_dl_iter)
            images_U = images_U.to(device)
            images_X, labels_X = next(labeled_trn_dl_iter)
            images_X, labels_X = images_X.to(device), labels_X.to(device)
            images_X = images_X.float()
            images_U = images_U.float()
            labels_X = one_hot(labels_X, num_classes)

            batch_loss = mixmatch_train_step(model, images_X, labels_X, images_U, loss_fn, optimizer, batch_size, num_classes, K_transforms, device,  alpha, T, K, lambda_U)
            train_epoch_losses.append(batch_loss)
            batch_accuracy = accuracy(images_X, labels_X, model)
            train_epoch_accuracies.append(batch_accuracy)

        train_epoch_loss = np.array(train_epoch_losses).mean()
        train_epoch_accuracy = np.array(train_epoch_accuracies).mean()

    
        labeled_val_dl_iter = iter(labeled_val_dl)
        val_epoch_losses, val_epoch_accuracies = [], []
        for ix, batch in enumerate(labeled_val_dl_iter):
            images_val, labels_val = batch
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            images_val = images_val.float()
            # pdb.set_trace()
            val_loss = val_loss_fn(model(images_val), labels_val)
            val_epoch_losses.append(val_loss.item())
            val_accuracy = accuracy(images_val, labels_val, model)
            val_epoch_accuracies.append(val_accuracy)
            
        val_epoch_loss = np.mean(val_epoch_losses)
        val_epoch_acc = np.mean(val_epoch_accuracies)
                
        if val_epoch_acc >= max_val_accuracy:
            torch.save(model.state_dict(), os.path.join(savingPath, trial, "best_model.pth"))
            max_val_accuracy = val_epoch_acc

        print(f"Epoch {epoch + 1}/{n_epoch}, Training Loss: {train_epoch_loss:.4f}, Training Accuracy:{train_epoch_accuracy :.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc}")
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        result = np.vstack([np.array(train_losses), np.array(train_accuracies), np.array(val_losses), np.array(val_accuracies)])
        np.save(os.path.join(savingPath, trial, "mixmatch_model_loss.npy"), result)
        
    return train_losses, train_accuracies, val_losses, val_accuracies

def tensor_to_image(tensor):
    """
    tensor has shape [3,64,64]
    Convert a tensor to image
    """
    image = tensor.clone().detach().cpu()
    image = image.permute(1, 2, 0)
    return image


def print_image(model, images_X, labels_X, images_U, batch_size, num_classes, K_transforms, device,  alpha, T, K, lambda_U, trial, save_dir):
    """
    Save image
    """
    os.makedirs(save_dir, exist_ok=True)
    X_prime, p_X, U_prime, p_U = mixmatch.mixMatch(model, images_X, labels_X, images_U, batch_size, num_classes, K_transforms, device,  alpha, T, K, lambda_U, trial)
    for i, x in enumerate(X_prime):
        timestamp = int(time.time())
        X_prime_img = tensor_to_image(x)
        plt.imshow(X_prime_img)
        plt.tight_layout()
        X_prime_filename = f'X_prime_{i}_{timestamp}.png'
        # X_prime_img.save(os.path.join(save_dir, X_prime_filename))
        plt.savefig(os.path.join(save_dir, X_prime_filename), bbox_inches='tight')
        
    for i, u in enumerate(U_prime):
        timestamp = int(time.time())
        U_prime_img = tensor_to_image(u)
        plt.imshow(U_prime_img)
        plt.tight_layout()
        U_prime_filename = f'U_prime_{i}_{timestamp}.png'
        # U_prime_img.save(os.path.join(save_dir, U_prime_filename))
        plt.savefig(os.path.join(save_dir, U_prime_filename), bbox_inches='tight')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def main(args):

    T = float(args.temperature)
    K = int(args.k_augment)
    alpha = float(args.alpha)
    lambda_U = float(args.lambda_U)
    lr = float(args.lr)
    trial = args.trial

    model, _, optimizer = get_model(5, lr)
    loss_fn = mixmatch.loss
    val_loss_fn = nn.CrossEntropyLoss()

    n_epoch = 100
    batch_size = 16
    num_classes = 5

    transform_1 = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 180))
        ])

    transform_2 = transforms.Compose([
            # transforms.GaussianBlur(kernel = 5),
            AddGaussianNoise(0., 1.)
        ])

    K_transforms = [transform_1, transform_2]

    labeled_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/train/_annotations.coco.json')
    labeled_dataloader = custom_utils.get_labeled_data("LabeledData/train", labeled_annotations, batch_size, False)

    unlabeled_annotations = custom_utils.get_annotations('Climbing-Holds-and-Volumes-14/train/_annotations.coco.json')
    unlabeled_dataloader = custom_utils.get_unlabeled_data("UnlabeledData/train", unlabeled_annotations, batch_size, False)

    labeled_valid_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/valid/_annotations.coco.json')
    labeled_valid_dataloader = custom_utils.get_labeled_data("LabeledData/valid", labeled_valid_annotations, batch_size, False)

    labeled_test_annotations = custom_utils.get_annotations('ClimbingHoldDetection-15/test/_annotations.coco.json')
    labeled_test_dataloader = custom_utils.get_labeled_data("LabeledData/test", labeled_test_annotations, batch_size, False)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    savingPath = "model_checkpoints"
    if not os.path.exists(os.path.join(savingPath, trial)):
        os.makedirs(os.path.join(savingPath, trial))

    train_losses, train_accuracies, val_losses, val_accuracies = mixmatch_train(model, loss_fn, val_loss_fn, optimizer, n_epoch, labeled_dataloader, unlabeled_dataloader, labeled_valid_dataloader, device, batch_size, num_classes, K_transforms, savingPath,  alpha, T, K, lambda_U, trial)
    print(f"train losses is : {train_losses}")
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    