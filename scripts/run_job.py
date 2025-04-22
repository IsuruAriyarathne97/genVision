import argparse
import yaml
from train import train_model
from evaluate import evaluate_model

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    # Train the model
    model = train_model(
        data_dir=config['data_dir'],
        num_classes=config['num_classes'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    # Evaluate the model
    results = evaluate_model(
        model_path=config['model_path'],
        data_dir=config['data_dir'],
        num_classes=config['num_classes'],
        batch_size=config['batch_size']
    )
    print("Evaluation Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)