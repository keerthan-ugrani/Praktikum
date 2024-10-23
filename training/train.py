# training/train_classification.py

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.cnn_model import CNNClassifier
from sklearn.metrics import accuracy_score

def train_cnn(train_data, test_data, config, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data
    input_shape = (1, config['target_image_size'][0], config['target_image_size'][1])
    num_classes = len(train_data)

    # Convert the train and test data to DataLoader format
    train_dataset = create_dataset(train_data, num_classes)
    test_dataset = create_dataset(test_data, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = CNNClassifier(input_shape, num_classes).to(device)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {running_loss / len(train_loader)}')

    # Evaluate the model
    accuracy = evaluate_cnn(model, test_loader, device)
    return model, accuracy

def evaluate_cnn(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'CNN classification accuracy: {accuracy * 100:.2f}%')
    return accuracy

def create_dataset(data, num_classes):
    """Creates a TensorDataset for training and testing."""
    images = []
    labels = []
    for class_idx, (class_name, class_images) in enumerate(data.items()):
        images.extend(class_images)
        labels.extend([class_idx] * len(class_images))
    
    # Convert to a NumPy array first for efficiency
    images_np = np.array(images)  # Convert list of numpy arrays to a single numpy array
    images_tensor = torch.tensor(images_np).float().unsqueeze(1)  # Add channel dimension for grayscale
    labels_tensor = torch.tensor(labels).long()
    return TensorDataset(images_tensor, labels_tensor)
