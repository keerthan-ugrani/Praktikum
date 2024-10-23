# training/train_few_shot.py

import torch
import torch.optim as optim
from models.few_shot_model import PrototypicalNetwork

def train_few_shot(support_data, query_data, config, loss_fn=None, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate input_shape dynamically
    target_image_size = config['target_image_size']
    input_shape = (1, target_image_size[0], target_image_size[1])  # (channels, height, width)
    n_classes = len(support_data)

    model = PrototypicalNetwork(input_shape, embedding_dim=config['embedding_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Find the minimum number of support samples across classes
    min_support_samples = min(len(support_data[class_name]) for class_name in support_data)
    print(f"Using {min_support_samples} support samples per class to ensure consistency.")

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0

        # List to hold all support embeddings for computing prototypes
        support_embeddings_list = []
        for class_name in support_data:
            # Use only the minimum number of support samples to ensure consistent shape
            support_samples = support_data[class_name][:min_support_samples]
            support = torch.tensor(support_samples).float().unsqueeze(1).to(device)  # Add channel dim
            support_embeddings = model(support)
            support_embeddings_list.append(support_embeddings)

        # Concatenate support embeddings from all classes
        support_embeddings = torch.cat(support_embeddings_list, dim=0)

        # Compute prototypes outside the query loop to avoid recomputation
        prototypes = model.compute_prototypes(support_embeddings, n_classes=n_classes)

        for class_name in query_data:
            query = torch.tensor(query_data[class_name]).float().unsqueeze(1).to(device)  # Add channel dim
            query_labels = torch.arange(len(query)).to(device)
            query_embeddings = model(query)

            # Compute distances between query embeddings and prototypes
            distances = model.euclidean_distance(query_embeddings, prototypes)

            # Calculate loss
            if loss_fn:
                loss = loss_fn(query_labels, -distances)
            else:
                loss = torch.nn.CrossEntropyLoss(weight=class_weights)(-distances, query_labels)

            # Perform a single backward pass and update the model
            optimizer.zero_grad()
            loss.backward()  # No need for retain_graph=True unless using multiple backward passes
            optimizer.step()

            # Detach tensors to prevent retaining the graph unnecessarily
            query_embeddings = query_embeddings.detach()
            support_embeddings = support_embeddings.detach()
            prototypes = prototypes.detach()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{config["num_epochs"]}, Loss: {total_loss / len(support_data)}')

    return model

def evaluate_few_shot(model, query_data, prototypes, config):
    """
    Evaluates the few-shot model using precomputed prototypes.
    
    Args:
        model (PrototypicalNetwork): The trained Prototypical Network.
        query_data (dict): Query data for evaluation.
        prototypes (torch.Tensor): Precomputed prototypes for each class.
        config (dict): Configuration dictionary.
    
    Returns:
        float: Accuracy of the model on the query set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for class_name, queries in query_data.items():
            # Prepare query embeddings
            queries = torch.tensor(queries).float().unsqueeze(1).to(device)  # Add channel dimension
            query_labels = torch.arange(len(queries)).to(device)

            # Compute query embeddings using the trained model
            query_embeddings = model(queries)

            # Compute distances between query embeddings and precomputed prototypes
            distances = model.euclidean_distance(query_embeddings, prototypes)
            predictions = torch.argmin(distances, dim=1)

            # Compare predictions with true labels
            correct += (predictions == query_labels).sum().item()
            total += query_labels.size(0)

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f'Few-shot learning evaluation accuracy: {accuracy * 100:.2f}%')
    return accuracy
