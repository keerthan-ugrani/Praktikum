from base_dataset_manager import BaseDatasetManager
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class DatasetManagerWithClassWeighting(BaseDatasetManager):
    def compute_class_weights(self):
        class_counts = self.metadata['class_counts']
        labels = np.concatenate([[class_name] * count for class_name, count in class_counts.items()])
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        return dict(enumerate(class_weights))
