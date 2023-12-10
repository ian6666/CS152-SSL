import torch

def sharpen(p, T):
    """
    p: prediction probabilities, [Batch, classes]
    T: temperature
    """
    pass
    # return p_i**(1/T)/(sum(p_j**(1/T)))

def mixUp(image_1, label_1, image_2, label_2, alpha):
    """
    image_1 is a dataset, Tensor
    label_1 is a dataset, Tensor

    returns
    mixed_images, mixed_labels
    """

        # def augment(self, x, l, beta, **kwargs):
        # del kwargs
        # mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1, 1])
        # mix = tf.maximum(mix, 1 - mix)
        # xmix = x * mix + x[::-1] * (1 - mix)
        # lmix = l * mix[:, :, 0, 0] + l[::-1] * (1 - mix[:, :, 0, 0])
        # return xmix, lmix
    pass


def mixMatch(model, X_dataloader_iter, U_dataloader_iter,  batch_size, num_classes, alpha=0.75, T=0.5, K=2):
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
    images_X, labels_X = next(X_dataloader_iter)
    images_U= next(U_dataloader_iter)

    # torch.random(dim)
    raw_labels_U = model(images_U) # [batch*k, classes]
    raw_labels_U_reshape = raw_labels_U.reshape([batch_size, K, num_classes])
    avg_labels_U = torch.mean(raw_labels_U_reshape, axis= 1) # check, average over K
    labels_U = sharpen(avg_labels_U, T) # [batch, classes]
    labels_U_reshape = torch.repeat_interleave(labels_U, K) # to expand back to [batch*k, classes]
    
    images_W_raw = torch.concat([images_X, images_U])
    labels_W_raw = torch.concat([labels_X, labels_U_reshape])
    shuffled_indices = torch.randperm(torch.arrange(len(images_W_raw)))
    images_W = images_W_raw[shuffled_indices]
    labels_W = labels_W_raw[shuffled_indices]
    X_prime = mixUp(images_X,labels_X,  images_W[:len(images_X)], labels_W[:len(images_X)], alpha)
    U_prime = mixUp(images_U, labels_U, images_W[len(images_X):], labels_W[len(images_X):], alpha)
    return X_prime, U_prime
    # 
    #     pass    # 


#TODO add matchmatch and loss function to training