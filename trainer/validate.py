import torch
from torch.utils.data import DataLoader
import wandb 
from tqdm import tqdm
import os

# Other imports
from utils.losses import compute_loss
from utils.metrics import compute_accuracy

# Utils imports
from utils.visualize import visualize

# Get the normalization constant for the loss
loss_norm_val = None

def validate(
    model, 
    data_loader, 
    loss_choice='L2',
    device='cpu',
    use_wandb=False,
    load_checkpoint=False,
    checkpoint_path=None,
    mode:str='eval',
    config: dict=None,
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
    global loss_norm_val
    
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
    epoch_loss = 0.0
    accuracy = []
    raw_losses = []  # List to store raw loss values from each batch.
    
    # Iterate over the data loader
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Batch", leave=False):
            
            # Get data and move to device
            description, query = data['summary'], data['query']
            gt_target, feature_map = data['target'], data['feature_map']
            # Move target and feature map to device
            gt_target, feature_map = gt_target.to(device), feature_map.to(device)

            # Forward pass
            output = model(description=description, map_tensor=feature_map, query=query)

            # Compute loss
            loss = compute_loss(gt_target, output, loss_choice, device)
            val_loss = loss.item()
            raw_losses.append(val_loss)
            epoch_loss += val_loss

            # Compute accuracy
            accuracy.append(compute_accuracy(gt_target, output))
            
            # Visualize results
            if config.visualize:
                for query_, gt_target_, value_map_, map_path_ in zip(query, gt_target, output['value_map'], data['map_path']):
                    visualize(
                        query_, 
                        gt_target_, 
                        value_map_, 
                        map_path_, 
                        batch_idx, 
                        name="prediction", 
                        split="val",
                        use_obstacle_map=config.use_obstacle_map,
                        upscale_factor=2.0
                    )
        
            if config.debug and batch_idx == 2:
                break
    
    # Compute average validation normalized loss
    loss_norm_val = sum(raw_losses) / len(raw_losses) if loss_norm_val is None else loss_norm_val
    val_avg_loss = sum(raw_losses) / len(raw_losses) / loss_norm_val
    
    # Calculate average accuracy for the epoch for each th key
    val_avg_acc = {key: sum(d[key] for d in accuracy) / len(accuracy) for key in accuracy[0]}
        
    # If evaluation mode log the results
    if mode in ['eval']:
        print(f"Val Loss: {val_avg_loss:.4f}")
        for key in val_avg_acc:
            print(f"\nVal Acc [{key}]: {val_avg_acc[key]:.4f}")
        print('-' * 20)
        
        # Log metrics to W&B if in evaluation mode
        if use_wandb:
            log_epoch_metrics(val_avg_loss, val_avg_acc)

    return val_avg_loss, val_avg_acc


# TODO: Since there is already a log_epoch_metrics function in utils/utils.py
# We can remove this function and make it cleaner
def log_epoch_metrics(val_loss, val_acc):
    metrics = {
        "Val Loss": val_loss,
    }
    metrics.update({f"Val Acc [{k}]": v for k, v in val_acc.items()})
    wandb.log(metrics)
    return