import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

def plot_loss(history, title="Training and Validation Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_metrics_comparison(metrics, methods, title="Comparison of Balancing Methods"):
    x = np.arange(len(methods))  # Label locations
    width = 0.25  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel('Methods')
    ax.set_title(title)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    ax.legend()
    plt.show()

def plot_gan_loss(d_losses, g_losses, title="GAN Training Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalize for percentage
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(title)
    plt.show()

def plot_f1_score(history, y_val, x_val, model, title="F1-Score Over Epochs"):
    f1_scores = []
    for epoch in range(len(history.history['loss'])):
        y_pred = np.argmax(model.predict(x_val), axis=1)
        f1 = f1_score(y_val, y_pred, average='weighted')
        f1_scores.append(f1)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, label='F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_ensemble_performance(method_scores, ensemble_score, title="Ensemble vs Individual Methods"):
    methods = list(method_scores.keys())
    scores = list(method_scores.values())
    methods.append("Ensemble")
    scores.append(ensemble_score)

    plt.figure(figsize=(10, 5))
    plt.bar(methods, scores, color=['blue'] * len(method_scores) + ['green'])
    plt.xlabel('Methods')
    plt.ylabel('F1-Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
