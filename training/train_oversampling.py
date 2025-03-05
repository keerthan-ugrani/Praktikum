from models.cnn_classifier import CNNClassifier
from dataset_managers.oversampling_manager import OversamplingManager

def train_oversampling(root_dir, x_train, y_train, x_val, y_val):
    # Apply oversampling
    oversampling_manager = OversamplingManager(root_dir)
    oversampling_manager.oversample()

    # Initialize and train CNN
    classifier = CNNClassifier()
    model = classifier.create_model()
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
    
    return model
