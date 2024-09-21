import argparse
import yaml
from dataset_managers.oversampling_manager import DatasetManagerWithOversampling
from dataset_managers.class_weight_manager import DatasetManagerWithClassWeighting
from dataset_managers.focal_loss_manager import DatasetManagerWithFocalLoss
from dataset_managers.gan_manager import DatasetManagerWithGANs
from training.train import train_model
from models.model import create_model

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Handle data imbalance methods and train models.")
    parser.add_argument('--method', type=str, required=True, choices=['oversampling', 'class_weight', 'focal_loss', 'gan'],
                        help="Select data imbalance handling method.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    data_dir = config['data']['data_dir']
    
    # Select the appropriate dataset manager
    if args.method == 'oversampling':
        dataset_manager = DatasetManagerWithOversampling(data_dir)
    elif args.method == 'class_weight':
        dataset_manager = DatasetManagerWithClassWeighting(data_dir)
    elif args.method == 'focal_loss':
        dataset_manager = DatasetManagerWithFocalLoss(data_dir)
    elif args.method == 'gan':
        dataset_manager = DatasetManagerWithGANs(data_dir)

    # Load and preprocess data
    dataset_manager.load_data()

    # Prepare the dataset based on selected method
    if args.method == 'oversampling':
        dataset_manager.oversample_data(config['oversampling']['target_class_size'])
    elif args.method == 'gan':
        dataset_manager.train_gan(config['gan'])

    # Split data into train/test sets
    train_data, test_data = dataset_manager.split_data(test_size=config['training']['test_size'])

    # Create model
    model = create_model(config['model'])

    # Train and evaluate the model
    train_model(model, train_data, test_data, config, dataset_manager)

if __name__ == '__main__':
    main()
