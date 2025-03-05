from utils.plotting import (
    plot_loss, 
    plot_metrics_comparison, 
    plot_gan_loss, 
    plot_confusion_matrix, 
    plot_f1_score, 
    plot_ensemble_performance
)
import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from training.train_class_weighted import train_class_weighted
from training.train_focal_loss import train_focal_loss
from training.train_oversampling import train_oversampling
from training.train_gan_augmentation import train_gan_augmentation
from models.cnn_classifier import CNNClassifier
import yaml

def remap_labels(y_train, y_val):
    """Remap labels to a zero-based range."""
    unique_labels = np.unique(np.concatenate([y_train, y_val]))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_mapping[label] for label in y_train])
    y_val = np.array([label_mapping[label] for label in y_val])
    print(f"Label mapping: {label_mapping}")
    return y_train, y_val, len(unique_labels)

def validate_labels(y_train, y_val, num_classes):
    """Ensure all labels are within the valid range."""
    assert np.all((y_train >= 0) & (y_train < num_classes)), "y_train contains invalid labels."
    assert np.all((y_val >= 0) & (y_val < num_classes)), "y_val contains invalid labels."

def load_data(root_dir, target_size=(128, 128)):
    # Replace this with your actual data loading logic
    x_train = np.random.rand(100, *target_size, 1)
    y_train = np.random.randint(0, 30, 100)  # Simulating invalid labels
    x_val = np.random.rand(20, *target_size, 1)
    y_val = np.random.randint(0, 30, 20)  # Simulating invalid labels
    return x_train, y_train, x_val, y_val

def main():
    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    root_dir = config["root_dir"]
    target_size = tuple(config["target_size"])
    latent_dim = config.get("latent_dim", 100)

    # Step 1: Load data
    x_train, y_train, x_val, y_val = load_data(root_dir, target_size)

    # Step 2: Remap and validate labels
    y_train, y_val, num_classes = remap_labels(y_train, y_val)
    validate_labels(y_train, y_val, num_classes)

    # Step 3: Train GAN for augmentation
    print("Training DCGAN for augmentation...")
    gan_model = train_gan_augmentation(root_dir, x_train, latent_dim=latent_dim)

    # Plot GAN training loss
    if gan_model and hasattr(gan_model, "d_losses") and hasattr(gan_model, "g_losses"):
        plot_gan_loss(gan_model.d_losses, gan_model.g_losses)
    else:
        print("GAN model did not return loss information. Skipping GAN loss plot.")

    # Step 4: Train models with different methods
    model_scores = {}

    # Class Weighted Training
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    model_class_weighted = train_class_weighted(root_dir, x_train, y_train, x_val, y_val, class_weight_dict)
    y_val_pred = np.argmax(model_class_weighted.predict(x_val), axis=1)
    model_scores["Class Weighted"] = (model_class_weighted, f1_score(y_val, y_val_pred, average="weighted"))

    # Focal Loss Training
    model_focal_loss = train_focal_loss(x_train, y_train, x_val, y_val, num_classes=num_classes)
    y_val_pred = np.argmax(model_focal_loss.predict(x_val), axis=1)
    model_scores["Focal Loss"] = (model_focal_loss, f1_score(y_val, y_val_pred, average="weighted"))

    # Oversampling
    model_oversampling = train_oversampling(root_dir, x_train, y_train, x_val, y_val)
    y_val_pred = np.argmax(model_oversampling.predict(x_val), axis=1)
    model_scores["Oversampling"] = (model_oversampling, f1_score(y_val, y_val_pred, average="weighted"))

    # GAN-Augmented Training
    print("Training with GAN-Augmented Data...")
    x_train_augmented, y_train_augmented, _, _ = load_data(root_dir, target_size)
    y_train_augmented, y_val, num_classes = remap_labels(y_train_augmented, y_val)
    gan_cnn_model = CNNClassifier(input_shape=target_size + (1,), num_classes=num_classes).create_model()
    history = gan_cnn_model.fit(x_train_augmented, y_train_augmented, epochs=10, batch_size=32, validation_data=(x_val, y_val))
    plot_loss(history, title="GAN-Augmented Training Loss")
    y_val_pred = np.argmax(gan_cnn_model.predict(x_val), axis=1)
    model_scores["GAN Augmentation"] = (gan_cnn_model, f1_score(y_val, y_val_pred, average="weighted"))

    # Step 5: Ensemble
    print("Ensembling models...")
    models = [model[0] for model in model_scores.values()]
    def ensemble_predictions(models, x_val):
        predictions = [model.predict(x_val) for model in models]
        averaged_predictions = np.mean(predictions, axis=0)
        return np.argmax(averaged_predictions, axis=1)

    y_ensemble = ensemble_predictions(models, x_val)
    ensemble_score = f1_score(y_val, y_ensemble, average="weighted")
    model_scores["Ensemble"] = (None, ensemble_score)

    plot_ensemble_performance({k: v[1] for k, v in model_scores.items()}, ensemble_score)

    # Step 6: Find the best method and save the model
    best_method, (best_model, best_score) = max(model_scores.items(), key=lambda item: item[1][1])
    print(f"The best method is {best_method} with an F1 score of {best_score:.4f}")

    if best_model:
        best_model.save("best_model.h5")
        print(f"Best model saved as 'best_model.h5'.")
    else:
        print("Ensemble method selected. No single model to save.")

    # Step 7: Plot confusion matrix
    y_best_pred = y_val_pred if best_method != "Ensemble" else y_ensemble
    plot_confusion_matrix(y_val, y_best_pred, class_names=[f"Class {i}" for i in range(num_classes)])

if __name__ == "__main__":
    main()
