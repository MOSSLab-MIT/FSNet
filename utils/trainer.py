import numpy as np
import pickle
import time
import os 

# import wandb 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import json

from utils.optimization_utils import *
from utils.lbfgs import nondiff_lbfgs_solve_vec, hybrid_lbfgs_solve_vec
from models.neural_networks import MLP

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float64)


def load_instance(config):
    """Loads problem instance, data, and sets up save directory."""

    # Load data
    seed = config['seed']
    method = config['method']
    val_size = config['val_size']
    test_size = config['test_size']
    prob_type = config['prob_type']
    prob_name = config['prob_name']
    prob_size = config['prob_size']

    # Map problem types to their corresponding problem classes
    if prob_type == 'convex':
        problem_names = {
            'qp': QPProblem,
            'qcqp': QCQPProblem,
            'socp': SOCPProblem,
        }
    elif prob_type == 'nonconvex':
        problem_names = {
            'qp': nonconvexQPProblem,
            'qcqp': nonconvexQCQPProblem,
            'socp': nonconvexSOCPProblem,
        }
    elif prob_type == 'nonsmooth_nonconvex':
        problem_names = {
            'qp': nonsmooth_nonconvexQPProblem,
            'qcqp': nonsmooth_nonconvexQCQPProblem,
            'socp': nonsmooth_nonconvexSOCPProblem,
        }
    
    if prob_name not in problem_names:
        raise NotImplementedError(f"Problem type '{prob_type}_{prob_name}' not implemented")
    
    # Construct filepath using consistent pattern
    seed_data = 2025
    filepath = os.path.join(
        'datasets', 
        prob_type, 
        prob_name,
        f"random{seed_data}_{prob_name}_dataset_var{prob_size[0]}_ineq{prob_size[1]}_eq{prob_size[2]}_ex{prob_size[3]}"
    )
    
    # Load dataset
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    
    # Create problem instance using the appropriate class
    data = problem_names[prob_name](dataset, val_size, test_size, seed)

    data.device = DEVICE
    print("Running on: ", DEVICE)
    for attr in dir(data):
        var = getattr(data, attr)
        if torch.is_tensor(var):
            try:
                setattr(data, attr, var.to(DEVICE))
            except AttributeError:
                pass

    if config['ablation'] == True:
        result_save_dir = os.path.join('ablation_results', prob_type, prob_name, str(data), config['network'] + '_' + config['method'], 'dist_'+ str(config['FSNet']['dist_weight']) + '_diff_' + str(config['FSNet']['max_diff_iter']))
    else:
        result_save_dir = os.path.join('results', prob_type, prob_name, str(data), config['network'] + '_' + config['method'])

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    
    return data, result_save_dir


def create_model(data, method, config):
    """Creates and returns a neural network model."""
    
    hidden_dim = config['nn_para']["hidden_dim"]
    num_layers = config['nn_para']["num_layers"]
    network = config['network']
    dropout = config['nn_para']["dropout"]

    if network == 'MLP':
        if method == "DC3":
            out_dim = data.partial_vars.shape[0]
            model = MLP(data.xdim, hidden_dim, out_dim, num_layers=num_layers, dropout=dropout)
        else:
            model = MLP(data.xdim, hidden_dim, data.ydim, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model}")
    return model.to(DEVICE)

def train_net(data, method, config, save_dir):
    """Trains a neural network model for constrained optimization.

    Args:
        data: Data object with training/validation/test sets, scaling, objective, and constraints.
        method: Optimization method ("penalty", "adaptive_penalty", "FSNet", "DC3", "projection").
        config: Configuration dictionary with network, optimization, and method-specific parameters.
        save_dir: Directory to save the trained model.

    Returns:
        The trained neural network model.

    Raises:
        ValueError: If an unknown optimization method is specified.
    """

    # Initialize wandb
    # if config['ablation'] == False:
    #     run = wandb.init(project="lids-ml-optimization",  # your project name
    #                     config=config,                 # log your config as config
    #                     name=config.get('run_name', f"{method}_{config['seed']}_{config['prob_type']}_{config['prob_name']}_{config['network']}_{time.strftime('%Y%m%d_%H%M%S')}"),
    #                     save_code=False)              # optional: save a copy of your code
    # else:
    #     run = wandb.init(project="ablation_lids-ml-optimization",  # your project name
    #                     config=config,                 # log your config as config
    #                     name=config.get('run_name', f"{method}_{config['seed']}_{config['prob_type']}_{config['prob_name']}_{config['network']}_{config['FSNet']['dist_weight']}_dropout{config['nn_para']['dropout']}_diff{config['FSNet']['max_diff_iter']}_{time.strftime('%Y%m%d_%H%M%S')}"),
    #                     save_code=False)              # optional: save a copy of your code
    
    # train_loader = DataLoader(data.train_dataset, batch_size=config['batch_size'], shuffle=True,
    #                           num_workers=2, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(data.train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(data.val_dataset, batch_size=config['val_size'], shuffle=False)
    train_size = len(data.train_dataset)
        
    # Extract weights for different methods
    # General
    num_epochs = config['nn_para']["num_epochs"]
    obj_weight = config[method].get('obj_weight', 0)
    eq_pen_weight = config[method].get('eq_pen_weight', 0)
    ineq_pen_weight = config[method].get('ineq_pen_weight', 0) 
    # for adaptive penalty method
    if method == 'adaptive_penalty':
        increasing_rate = config['adaptive_penalty']['increasing_rate']
        eq_pen_weight_max = config['adaptive_penalty']['eq_pen_weight_max']
        ineq_pen_weight_max = config['adaptive_penalty']['ineq_pen_weight_max']

    if method == 'projection':
        dist_weight = config[method].get('dist_weight', 1)

    if method == 'FSNet':
        val_tol = config[method].get('val_tol', 0)
        max_iter = config[method].get('max_iter', 0)
        max_diff_iter = config[method].get('max_diff_iter', 0)
        memory_size = config[method].get('memory_size', 0)
        decreasing_tol_step = int(config['FSNet']['decreasing_tol_step'])
        scale = config['FSNet'].get('scale', 1)
        dist_weight = config[method].get('dist_weight', 1)
        
    
    # create model
    model = create_model(data, method, config)
    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['nn_para']['lr'], weight_decay=0.001, fused=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['nn_para']['lr_decay_step'], gamma=config['nn_para']['lr_decay'])

    losses = []
    distance = 0.0
    train_start = time.time()

    # Training loop
    for i in range(num_epochs):
        loss_epoch = 0.0
        obj_epoch = 0.0
        eq_violation_epoch = 0.0
        ineq_violation_epoch = 0.0
        start_time = time.time()
        # Training loop
        # use this if you want to decrease the tolerance for FSNet during training. This helps speed up training!
        if method =='FSNet' and (i+1) % decreasing_tol_step == 0:
            val_tol = np.clip(val_tol/10, a_min=1.0e-9, a_max=1.0e-6) 
            config['FSNet']['val_tol'] = val_tol


        # decrease dropout rate during training
        if i == 100:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = m.p/2
        elif i==150:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0

        model.train()
        for (X_batch, _) in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)       
            Y_pred = model(X_batch)
            Y_pred_scaled = data.scale(Y_pred) # Scale the output to the original range

            if method == "penalty":
                Y_post = Y_pred_scaled
                obj = data.obj_fn(Y_post)
                eq_violation = (data.eq_resid(X_batch, Y_post)**2).sum(dim=1)
                ineq_violation = (data.ineq_resid(X_batch, Y_post)**2).sum(dim=1)
                loss = obj_weight*obj + eq_pen_weight*eq_violation + ineq_pen_weight*ineq_violation
            

            elif method == "adaptive_penalty":
                Y_post = Y_pred_scaled
                obj = data.obj_fn(Y_post)
                eq_violation = (data.eq_resid(X_batch, Y_post)**2).sum(dim=1)
                ineq_violation = (data.ineq_resid(X_batch, Y_post)**2).sum(dim=1)
                loss = obj_weight*obj + eq_pen_weight*eq_violation + ineq_pen_weight*ineq_violation
                
                # Adaptive penalty weightsX
                with torch.no_grad():
                    eq_pen_weight = torch.clamp(eq_pen_weight + increasing_rate * eq_violation.mean(), min=0.0, max=eq_pen_weight_max)
                    ineq_pen_weight = torch.clamp(ineq_pen_weight + increasing_rate * ineq_violation.mean(), min=0.0, max=ineq_pen_weight_max)
                    # Reset the weights if the violation is small
                    if eq_pen_weight >= eq_pen_weight_max:
                        eq_pen_weight = eq_pen_weight_max/2
                    if ineq_pen_weight >= ineq_pen_weight_max:
                        ineq_pen_weight = ineq_pen_weight_max/2
                # Log the adaptive penalty weights
                # wandb.log({
                #     "eq_penalty_weight": eq_pen_weight,
                #     "ineq_penalty_weight": ineq_pen_weight
                # })


            elif method == "FSNet":
                Y_post = hybrid_lbfgs_solve_vec(Y_pred_scaled, X_batch, data, val_tol=val_tol, memory=memory_size, max_iter=max_iter, max_diff_iter=max_diff_iter, scale=scale)
                obj = data.obj_fn(Y_post)
                pre_eq_violation = (data.eq_resid(X_batch, Y_pred_scaled)**2).sum(dim=1)
                pre_ineq_violation = (data.ineq_resid(X_batch, Y_pred_scaled)**2).sum(dim=1)
                # compute the distance between the predicted and the feasible solution
                distance = (torch.norm(Y_post - Y_pred_scaled, dim=1)**2).mean()
                if pre_eq_violation.mean() >= 1e3 or pre_ineq_violation.mean() >= 1e3:
                    loss = obj_weight*obj + dist_weight*distance + eq_pen_weight*pre_eq_violation + ineq_pen_weight*pre_ineq_violation
                else:
                    loss = obj_weight*obj + dist_weight*distance
                

            elif method == "DC3":
                Y_pred_scaled = data.complete_partial(X_batch, Y_pred_scaled) # Complete the solution - completion step
                Y_post = grad_steps(data, X_batch, Y_pred_scaled, config) # Unroll the correction steps
                obj = data.obj_fn(Y_post)
                eq_violation = (data.eq_resid(X_batch, Y_post)**2).sum(dim=1)
                ineq_violation = (data.ineq_resid(X_batch, Y_post)**2).sum(dim=1)
                loss = obj_weight*obj + eq_pen_weight*eq_violation + ineq_pen_weight*ineq_violation


            elif method == "projection":
                Y_post = data.qpth_projection(X_batch, Y_pred_scaled)
                obj = data.obj_fn(Y_post)
                eq_violation = (data.eq_resid(X_batch, Y_post)**2).sum(dim=1)
                ineq_violation = (data.ineq_resid(X_batch, Y_post)**2).sum(dim=1)
                # compute the distance between the predicted and the feasible solution
                distance = (torch.norm(Y_post - Y_pred_scaled, dim=1)**2).mean()
                loss = obj_weight*obj + dist_weight*distance

            else:
                raise ValueError(f"Unknown method: {method}")
            

            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            #compute abs violations for logging and printing
            eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
            ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
            with torch.no_grad():
                loss_epoch += loss.sum().item()
                obj_epoch += obj.sum().item()
                eq_violation_epoch += eq_violation.sum().item()
                ineq_violation_epoch += ineq_violation.sum().item()
        
        # Compute the average by dividing by total_samples
        loss_epoch /= train_size
        obj_epoch /= train_size
        eq_violation_epoch /= train_size
        ineq_violation_epoch /= train_size
        
        losses.append(loss_epoch) 
        # Log training metrics
        # wandb.log({
        #     "epoch": i,
        #     "train/loss": loss_epoch,
        #     "train/objective": obj_epoch,
        #     "train/eq_violation": eq_violation_epoch,
        #     "train/ineq_violation": ineq_violation_epoch,
        #     "train/distance": distance,
        #     "lr": optimizer.param_groups[0]['lr'],
        #     "epoch_time": time.time() - start_time
        # })

        print(f"Epoch {i+1}/{num_epochs}, Loss: {loss_epoch:.4f}, Obj: {obj_epoch:.4f}, eq_violation: {eq_violation_epoch:.5f}, ineq_violation: {ineq_violation_epoch:.5f}, Time: {time.time()-start_time:.2f}s")

        #Validation 
        if i % 50 == 0:
            model.eval()
            eval_results = evaluate_model(model, data, val_loader, method, config)
            
            # val_loss = eval_results["val_loss"]
            val_obj = eval_results["val_obj"]
            val_eq_violation = eval_results["val_eq_violation"]
            val_ineq_violation = eval_results["val_ineq_violation"]
            opt_gap = eval_results["opt_gap"]
            
            print(f"Epoch {i+1}/{num_epochs}, Loss: {loss_epoch:.4f}, Obj: {obj_epoch:.4f}, eq_violation: {eq_violation_epoch:.5f}, ineq_violation: {ineq_violation_epoch:.5f}, Time: {time.time()-start_time:.2f}s")
            print(f"Validation: val_obj: {val_obj.mean().item():.4f}, opt_gap: {opt_gap.mean().item():.5f} +min: {opt_gap.min().item():.5f} +max: {opt_gap.max().item():.5f}, val_eq_violation: {val_eq_violation.mean().item():.5f} +max: {val_eq_violation.max().item():.5f}, val_ineq_violation: {val_ineq_violation.mean().item():.5f}, +max:{val_ineq_violation.max().item():.5f}")
            
           
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_seed{config['seed']}_{time.strftime('%Y%m%d_%H%M%S')}.pt"))
    
    training_time = time.time() - train_start
    # wandb.log({"train/training_time": training_time})
    # test
    print("Evaluating model on test data")
    for batch_size in {config['test_size']}:
        test_solver_net(model, data, method, config, save_dir, batch_size=batch_size)
    
    # wandb.finish()
    return model

#Evaluation for trained mododel
def evaluate_model(model, data, val_loader, method, config):
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Neural network model
        data: Problem data instance
        val_loader: DataLoader for validation data
        method: Optimization method ("penalty", "FSNet", "DC3", "projection")
        config: Dictionary of arguments and parameters    
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    #Extract weights for different methods
    
    val_tol = config[method].get('val_tol', 0)
    max_iter = config[method].get('max_iter', 0)
    memory_size = config[method].get('memory_size', 0)
    scale = config['FSNet'].get('scale', 1)

    with torch.no_grad():
        for (X_batch, Y_batch) in val_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            Y_pred = model(X_batch)
            Y_pred_scaled = data.scale(Y_pred)
            

            if method == "penalty" or method == "adaptive_penalty":
                val_obj = data.obj_fn(Y_pred_scaled)
                val_eq_violation = data.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
                val_ineq_violation = data.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
            
            elif method == "FSNet":
                with torch.enable_grad():
                    Y_post = nondiff_lbfgs_solve_vec(Y_pred_scaled, X_batch, data, val_tol=val_tol, memory=memory_size, max_iter=max_iter, scale=scale)
                val_obj = data.obj_fn(Y_post)
                val_eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
                val_ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
                # compute the distance between the predicted and the feasible solution
                # val_distance = (torch.norm(Y_post - Y_pred_scaled, dim=1)**2).mean()
            
            elif method == "DC3":
                with torch.enable_grad():
                    Y_pred_scaled = data.complete_partial(X_batch, Y_pred_scaled)  # Complete the solution
                    Y_post = grad_steps(data, X_batch, Y_pred_scaled, config)  # Unroll correction steps
                val_obj = data.obj_fn(Y_post)
                val_eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
                val_ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
            
            elif method == "projection":
                with torch.enable_grad():
                    # Y_proj,  = data.cvx_projection(X_batch, Y_pred_scaled)
                    Y_post = data.qpth_projection(X_batch, Y_pred_scaled)

                val_obj = data.obj_fn(Y_post)
                val_eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
                val_ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
                # val_distance = (torch.norm(Y_post - Y_pred_scaled, dim=1)**2).mean()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            true_obj = data.obj_fn(Y_batch) 
            opt_gap = (val_obj - true_obj)/true_obj.abs()
            
            # Log to wandb
            # wandb.log({
            #     "val/opt_gap": opt_gap.mean().item(),
            #     "val/eq_violation": val_eq_violation.mean().item(),
            #     "val/ineq_violation": val_ineq_violation.mean().item()
            # })
            
            # We only process one batch since validation is typically done on a single batch
            return {
                "val_obj": val_obj,
                "val_eq_violation": val_eq_violation,
                "val_ineq_violation": val_ineq_violation,
                "opt_gap": opt_gap
            }


# Test the solver net
def test_solver_net(model, data, method, config, result_save_dir, batch_size):
    """
    Test trained model on test data and save results.
    
    Args:
        model: Neural network model
        data: Problem data instance
        method: Optimization method ("penalty", "FSNet", "DC3", "projection")
        config: Dictionary of arguments and parameters
        result_save_dir: Directory to save test results
        batch_size: Batch size for testing
        
    Returns:
        Dictionary containing test metrics
    """
    model.eval()
    test_loader = DataLoader(data.test_dataset, batch_size=batch_size, shuffle=False)
    val_tol = config[method].get('val_tol', 0)
    max_iter = config[method].get('max_iter', 0)
    memory_size = config[method].get('memory_size', 0)
    scale = config['FSNet'].get('scale', 1)

    test_eq_violations = []
    test_ineq_violations = []
    test_objs = []
    true_objs = []
    test_gaps = []

    with torch.no_grad():
        start_time = time.time()
        for (X_batch, Y_batch) in test_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            Y_pred = model(X_batch)
            Y_pred_scaled = data.scale(Y_pred)
            
            if method == "penalty" or method == "adaptive_penalty":
                test_obj = data.obj_fn(Y_pred_scaled)
                test_eq_violation = data.eq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
                test_ineq_violation = data.ineq_resid(X_batch, Y_pred_scaled).abs().sum(dim=1)
            
            elif method == "FSNet":
                with torch.enable_grad():
                    Y_post = nondiff_lbfgs_solve_vec(Y_pred_scaled, X_batch, data, val_tol=val_tol, memory=memory_size, max_iter=max_iter, scale=scale)
                test_obj = data.obj_fn(Y_post)
                test_eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
                test_ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
            
            elif method == "DC3":
                with torch.enable_grad():
                    Y_pred_scaled = data.complete_partial(X_batch, Y_pred_scaled)
                    Y_post = grad_steps(data, X_batch, Y_pred_scaled, config)
                test_obj = data.obj_fn(Y_post)
                test_eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
                test_ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
            
            elif method == "projection":
                with torch.enable_grad():
                    Y_post = data.qpth_projection(X_batch, Y_pred_scaled)

                test_obj = data.obj_fn(Y_post)
                test_eq_violation = data.eq_resid(X_batch, Y_post).abs().sum(dim=1)
                test_ineq_violation = data.ineq_resid(X_batch, Y_post).abs().sum(dim=1)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            true_obj = data.obj_fn(Y_batch)
            opt_gap = (test_obj - true_obj)/true_obj.abs()

            # Append results to lists
            test_objs.append(test_obj.cpu().detach().numpy())
            true_objs.append(true_obj.cpu().detach().numpy())
            test_eq_violations.append(test_eq_violation.cpu().detach().numpy())
            test_ineq_violations.append(test_ineq_violation.cpu().detach().numpy())
            test_gaps.append(opt_gap.cpu().detach().numpy())

        end_time = time.time()
        # Calculate optimality gap
        
        # Log metrics to wandb
        # wandb.log({
        #     "test/opt_gap": opt_gap.mean().item(),
        #     "test/eq_violation": test_eq_violation.mean().item(),
        #     "test/ineq_violation": test_ineq_violation.mean().item(),
        #     "test/raw_time": end_time - start_time
        # })
            
        print(f"Test: test_obj: {test_obj.mean().item():.4f}, opt_gap: {opt_gap.mean().item():.5f} +min: {opt_gap.min().item():.5f} +max: {opt_gap.max().item():.5f}, "
            f"test_eq_violation: {test_eq_violation.mean().item():.5f} +max: {test_eq_violation.max().item():.5f}, test_ineq_violation: {test_ineq_violation.mean().item():.5f} +max: {test_ineq_violation.max().item():.5f}")
            
        # Save the results to a file
        result_save_path = os.path.join(result_save_dir, f"test_results_seed{config['seed']}_batch{batch_size}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        # Convert lists of arrays to single numpy arrays
        test_objs_array = np.concatenate(test_objs)
        true_objs_array = np.concatenate(true_objs)
        test_gaps_array = np.concatenate(test_gaps)
        test_eq_violations_array = np.concatenate(test_eq_violations)
        test_ineq_violations_array = np.concatenate(test_ineq_violations)
        
        # Print latest batch results for logging purposes
        print(f"Test (batch={batch_size}): Completed {len(test_objs_array)} test samples in {end_time - start_time:.3f} seconds")
        
        # Save the results to a file
        result_save_path = os.path.join(result_save_dir, f"test_results_seed{config['seed']}_batch{batch_size}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(result_save_path, 'w') as f:
            json.dump({
            "test_obj": test_objs_array.tolist(),
            "true_obj": true_objs_array.tolist(),
            "opt_gap": test_gaps_array.tolist(),
            "test_eq_violation": test_eq_violations_array.tolist(),
            "test_ineq_violation": test_ineq_violations_array.tolist(),
            "test_time": end_time - start_time,
            "batch_size": batch_size
            }, f, indent=4)

    return {
        "test_obj": test_obj,
        "opt_gap": opt_gap,
        "test_eq_violation": test_eq_violation, 
        "test_ineq_violation": test_ineq_violation
    }