from models.cnn_classifier import CNNClassifier
from dataset_managers.focal_loss_manager import focal_loss

def train_focal_loss(x_train, y_train, x_val, y_val):
    classifier = CNNClassifier()
    model = classifier.create_model()

    model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))
    return model
