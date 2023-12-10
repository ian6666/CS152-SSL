import torchvision
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms,models,datasets
from custom_utils import augmentAndPredict
import matplotlib.pyplot as plt
import pdb

def sharpen(p, T):
    """
    p: prediction probabilities, [Batch, classes]
    T: temperature
    returns
    normalized, [Batch, classes]
    """
    powered_p = p.pow(1/T)
    row_sum = powered_p.sum(dim = 1, keepdim = True)
    normalized = p/row_sum
    return normalized
    

def mixUp(image_1, label_1, image_2, label_2, alpha):
    """
    image_1 is a dataset, Tensor
    label_1 is a dataset, Tensor

    returns
    mixed_images, mixed_labels
    """
    l = torch.distributions.beta.Beta(alpha, alpha).sample()
    l = max(l , 1 - l)
    x = l * image_1 + (1 - l) * image_2
    p = l * label_1 + (1 - l) * label_2
    return x, p


def mixMatch(model, images_X, labels_X, images_U, batch_size, num_classes, K_transforms, device, alpha, T, K):
    """
    model: model
    iter(X_dataloader): [Batch, 3, 460, 460], [Batch, classes] <- labels
    iter(U_dataloader): might already account for K, returns [Batch*K, 3, 460, 460], _
    T: temperature for sharpening
    K: number of rounds of augmentation
    alpha: Beta distribution parameter for Mix
    batch_size: batch size
    num_classes: number of classes

    returns
    X_prime = (mixed_images, mixed_labels) ([batch, 460, 460], [batch, classes])
    U_prime = (mixed_images, mixed_labels)([batch*k, 460, 460], [batch*k, classes])
    """
    # images_X, labels_X = next(X_dataloader_iter)
    # # TODO change labels_X to be one hot
    # images_U= next(U_dataloader_iter)

    # torch.random(dim)
    # raw_labels_U = model(images_U) # [batch*k, classes]
    images_U_K, raw_labels_U = augmentAndPredict(model, images_U, K, num_classes, K_transforms, device)
    raw_labels_U_reshape = raw_labels_U
    images_U_K = images_U_K.reshape( (K*batch_size, images_U_K.shape[-3], images_U_K.shape[-2], images_U_K.shape[-1]) )
    # raw_labels_U_reshape = raw_labels_U.reshape([batch_size, K, num_classes])
    avg_labels_U = torch.mean(raw_labels_U_reshape, axis= 1) # check, average over K
    labels_U = sharpen(avg_labels_U, T) # [batch, classes]
    labels_U_reshape = torch.repeat_interleave(labels_U, K, axis=0) # to expand back to [batch*k, classes]
    images_W_raw = torch.concat([images_X, images_U_K])
    labels_W_raw = torch.concat([labels_X, labels_U_reshape])
    shuffled_indices = torch.randperm(len(images_W_raw))
    images_W = images_W_raw[shuffled_indices]
    labels_W = labels_W_raw[shuffled_indices]
    X_prime, p_X = mixUp(images_X, labels_X,  images_W[:len(images_X)], labels_W[:len(images_X)], alpha)
    U_prime, p_U = mixUp(images_U_K, labels_U_reshape, images_W[len(images_X):], labels_W[len(images_X):], alpha)
    return X_prime, p_X, U_prime, p_U

def loss(X_prime, U_prime, p_X, p_U, model, num_classes, lambda_U):
    """
    X_prime: labeled data
    U_prime: unlabeled data
    p_X: probability of X_prime
    p_U: probability of U_prime
    (input of this function is the output of mixmatch function)
    model: model
    num_classes: a constant (=5)
    """
    loss = nn.CrossEntropyLoss()
    L_X = 1/X_prime.shape[0] * loss(p_X,model(X_prime))
    L_U = 1/(num_classes * U_prime.shape[0]) * torch.sum((p_U - model(U_prime))**2)
    return L_X + L_U * lambda_U

