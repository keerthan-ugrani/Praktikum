project/
│
├── dataset/ # Folder to store the actual dataset
│ └── raw_data/ # Raw dataset (e.g., DICOM or other images)
│ └── processed_data/ # Folder to store processed HDF5 or preprocessed data
│
├── dataset_managers/ # Dataset management scripts for different methods
│ ├── **init**.py
│ ├── oversampling_manager.py # Oversampling and augmentation for minority classes
│ ├── class_weight_manager.py # Class-weighted loss handling
│ ├── focal_loss_manager.py # Focal loss implementation
│ ├── gan_manager.py # GAN-based data augmentation
│
├── models/ # Store your models and related scripts
│ ├── **init**.py
│ ├── model.py # Model creation script
│
├── training/ # Training-related scripts
│ ├── **init**.py
│ ├── train.py # Training loop and evaluation logic
│
├── main.py # Main entry point to select methods and run the pipeline
├── config.yaml # Configuration file with all hyperparameters and paths
├── requirements.txt # Python dependencies
