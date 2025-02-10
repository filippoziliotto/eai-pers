import torch
from torch.utils.data import DataLoader
import wandb 
from tqdm import tqdm
import os

# Other imports
from utils.losses import compute_loss
from utils.metrics import compute_accuracy

# Config file
import config
from utils.visualize import visualize

def log_epoch_metrics(val_loss, val_acc):
    metrics = {
        "Val Loss": val_loss,
    }
    metrics.update({f"Val Acc [{k}]": v for k, v in val_acc.items()})
    wandb.log(metrics)

def validate(
    model, 
    data_loader, 
    loss_choice='L2',
    device='cpu',
    use_wandb=False,
    load_checkpoint=False,
    checkpoint_path=None,
    **kwargs
    ):
    
    """
    Validates the model on the given data_loader and logs the validation loss to W&B.

    Args:
        model: The PyTorch model to validate.
        data_loader: DataLoader providing (ground_truth_coords, input_data) batches.
        loss_choice: Loss function choice ('L1' or 'L2').
        device: Device to run the validation on ('cpu' or 'cuda').
        use_wandb: If True, log metrics to W&B.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        val_avg_loss: Average validation loss.
        val_avg_acc: Average validation accuracy for each threshold.
    """
    
    # Move model to device if not already on it
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        model.to(device)
    
    # Load model weights from checkpoint
    if load_checkpoint:
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}...")
    
    # Set model to evaluation mode
    model.eval()
    val_loss = 0.0
    accuracy = []
    
    # Iterate over the data loader
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Batch", leave=False):
            
            # Get data and move to device
            description = data['description']
            query = data['query']
            gt_target = data['target'].to(device)
            feature_map = data['feature_map'].to(device)

            # Forward pass
            value_map = model(description=description, map_tensor=feature_map, query=query)

            # Compute loss
            loss, pred_target = compute_loss(gt_target, value_map, loss_choice, device)
            val_loss += loss.item()
            
            # Compute accuracy
            accuracy.append(compute_accuracy(gt_target, pred_target))
            
            # Visualize results
            if config.VISUALIZE:
                # Take the last input/output batch
                for query, gt_target, value_map, map_path in zip(query, gt_target, value_map, data['map_path']):
                    # Visualize the value_map
                    visualize(query, gt_target, value_map, map_path)
        
            if config.DEBUG and batch_idx == 2:
                break
    
    # Calculate average validation loss
    val_avg_loss = val_loss / len(data_loader)
    
    # accuracy = [ {'5': 0.8, '10': 0.9, '20': 1.0}, {'5': 0.8, '10': 0.9, '20': 1.0}, ...]
    val_avg_acc = {key: sum(d[key] for d in accuracy) / len(accuracy) for key in accuracy[0]}
    
    # Log metrics to W&B if in evaluation mode
    if use_wandb and kwargs.get('mode', 'eval'):
        log_epoch_metrics(val_avg_loss, val_avg_acc)

    return val_avg_loss, val_avg_acc