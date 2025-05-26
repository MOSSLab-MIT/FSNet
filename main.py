import yaml
import torch
import time
import argparse
from utils.trainer import load_instance, train_net

# Define available problem types and problems
PROBLEM_TYPES = ['convex', 'nonconvex', 'nonsmooth_nonconvex']
PROBLEM_NAMES = ['qp', 'qcqp', 'socp']

def create_parser():
    """Create and configure the argument parser, then load and process the configuration."""
    parser = argparse.ArgumentParser(description='Neural Network Optimization')
    
    # General parameters
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--method', type=str, 
                        help='Training method (penalty, adaptive_penalty, FSNet, DC3, projection)')
    parser.add_argument('--prob_type', type=str, choices=PROBLEM_TYPES,
                        help='Problem type (convex, nonconvex, nonsmooth_nonconvex)')
    parser.add_argument('--prob_name', type=str, choices=PROBLEM_NAMES,
                        help='Problem name (qp, qcqp, socp)')
    parser.add_argument('--prob_size', type=int, nargs='+', default=[100, 50, 50, 10000],
                        help='Problem size parameters [n, m, p, N] (default: [100, 50, 50, 10000])')
    parser.add_argument('--network', type=str, default='MLP',
                        help='Type of neural network to use')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--ablation', type=bool, default=False)

    # Dataset parameters
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--val_size', type=int, help='Size of validation dataset')
    parser.add_argument('--test_size', type=int, help='Size of test dataset')
    parser.add_argument('--dropout', type=float, help='Dropout rate for the model')

    # Neural network parameters
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', type=int, help='Learning rate decay step size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, help='Number of hidden layers')
    
    # Feasibility seeking parameters
    parser.add_argument('--scale', type=float, help='Scale')
    parser.add_argument('--dist_weight', type=float, help='Distance weight')
    parser.add_argument('--max_diff_iter', type=int, help='Maximum number of iterations for keeping the track of gradient')

    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments if provided
    if args.method:
        config['seed'] = args.seed
    if args.method:
        config['method'] = args.method
    if args.prob_type:
        config['prob_type'] = args.prob_type
    if args.prob_name:
        config['prob_name'] = args.prob_name
    if args.prob_size:
        config['prob_size'] = args.prob_size
    if args.network:
        config['network'] = args.network
    
    # Override dataset parameters
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.val_size:
        config['val_size'] = args.val_size
    if args.test_size:
        config['test_size'] = args.test_size
    
    # Override neural network parameters
    if args.lr:
        config['nn_para']['lr'] = args.lr
    if args.lr_decay:
        config['nn_para']['lr_decay'] = args.lr_decay
    if args.lr_decay_step:
        config['nn_para']['lr_decay_step'] = args.lr_decay_step
    if args.num_epochs:
        config['nn_para']['num_epochs'] = args.num_epochs
    if args.hidden_dim:
        config['nn_para']['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        config['nn_para']['num_layers'] = args.num_layers
    if args.dropout:
        config['nn_para']['dropout'] = args.dropout
    
    # Feasibility seeking parameters
    if args.scale:
        config['FSNet']['scale'] = args.scale
    if args.dist_weight is not None:
        config['FSNet']['dist_weight'] = args.dist_weight
    if args.max_diff_iter is not None:
        config['FSNet']['max_diff_iter'] = args.max_diff_iter

    # Ablation study flag
    config['ablation'] = args.ablation

    return args, config

def main():
    # Parse command-line arguments and get processed config
    args, config = create_parser()
    
    # Get the problem type and name from config (with defaults)
    prob_type = config.get('prob_type', 'Error')
    prob_name = config.get('prob_name', 'Error')
    
    print(f"\n======= Running for problem: {prob_type}/{prob_name} =======\n")

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Load data 
    print(f"Loading problem instance: {prob_type}/{prob_name} with size {config['prob_size']}")
    data, result_save_dir = load_instance(config)
    
    # Train and test the model
    print(f"Training model using {config['method']} method with seed {config['seed']} for {config['nn_para']['num_epochs']} epochs")
    start_time = time.time()
    train_net(data, config['method'], config, result_save_dir)
    training_time = time.time() - start_time
    print(f"Training and testing completed in {training_time:.2f} seconds")
    return print("Done!!!")

if __name__ == "__main__":
    main()