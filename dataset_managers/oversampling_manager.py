from .base_dataset_manager import BaseDatasetManager
import os
import shutil
import random

class OversamplingManager(BaseDatasetManager):
    def oversample(self):
        """Oversample classes to balance the dataset by duplicating images."""
        self.load_data()
        class_counts = self.metadata['class_name'].value_counts()
        max_count = class_counts.max()

        for class_name, count in class_counts.items():
            class_folder = os.path.join(self.root_dir, class_name)
            images_in_class = [f for f in os.listdir(class_folder) if f.endswith('.dcm') and not f.startswith("dup_")]
            
            # Duplicate images until the number of images matches the maximum class count
            duplicates_created = 0
            while len(images_in_class) + duplicates_created < max_count:
                img_to_duplicate = random.choice(images_in_class)
                src = os.path.join(class_folder, img_to_duplicate)
                
                # Generate a unique name for each duplicate to avoid conflicts
                dst = os.path.join(class_folder, f"dup_{duplicates_created}_{img_to_duplicate}")
                
                # Check if source file exists to avoid errors
                if os.path.isfile(src):
                    shutil.copy(src, dst)
                    duplicates_created += 1
                else:
                    print(f"Warning: Source file {src} does not exist.")
        
        print("Oversampling complete.")
