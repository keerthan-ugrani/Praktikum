from .base_dataset_manager import BaseDatasetManager
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class ClassWeightManager(BaseDatasetManager):
    def calculate_class_weights(self):
        self.load_data()
        
        # Map class names to integer labels
        class_names = self.metadata['class_name'].unique()
        class_name_to_label = {name: idx for idx, name in enumerate(class_names)}
        
        # Replace class names with integer labels in metadata
        labels = self.metadata['class_name'].map(class_name_to_label).values
        
        # Compute class weights using integer labels
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        
        # Return class weights as a dictionary mapping from integer labels
        return dict(enumerate(class_weights))
