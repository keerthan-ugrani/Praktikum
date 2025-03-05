from models.cnn_classifier import CNNClassifier
from dataset_managers.class_weight_manager import ClassWeightManager
from keras.utils import to_categorical

def train_class_weighted(root_dir, x_train, y_train, x_val, y_val):
    manager = ClassWeightManager(root_dir)
    class_weights = manager.calculate_class_weights()
    classifier = CNNClassifier()
    model = classifier.create_model()

    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val), class_weight=class_weights)
    return model
