# models/few_shot_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetwork(nn.Module):
    def __init__(self, input_shape, embedding_dim=64):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

    def compute_prototypes(self, support_embeddings, n_classes):
        """
        Computes prototypes for each class.
        Args:
            support_embeddings (tensor): Support set embeddings of shape (num_support_samples * n_classes, embedding_dim).
            n_classes (int): Number of classes.
        Returns:
            tensor: Prototypes for each class of shape (n_classes, embedding_dim).
        """
        num_support_samples_per_class = support_embeddings.size(0) // n_classes
        assert num_support_samples_per_class * n_classes == support_embeddings.size(0), \
            "Number of support samples is not evenly divisible by the number of classes."

        # Reshape support_embeddings to (n_classes, num_support_samples_per_class, embedding_dim)
        support_embeddings = support_embeddings.view(n_classes, num_support_samples_per_class, -1)
        
        # Compute the mean for each class to get the prototype
        prototypes = support_embeddings.mean(dim=1)
        return prototypes

    def euclidean_distance(self, x, y):
        """
        Computes the Euclidean distance between two tensors.
        Args:
            x (tensor): Query embeddings of shape (num_queries, embedding_dim).
            y (tensor): Prototype embeddings of shape (n_classes, embedding_dim).
        Returns:
            tensor: Distance matrix of shape (num_queries, n_classes).
        """
        # Ensure both x and y are at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        
        return torch.cdist(x, y, p=2)
