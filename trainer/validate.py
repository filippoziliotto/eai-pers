import torch
from torch.utils.data import DataLoader
import wandb  # Import W&B

# Other imports
from utils.losses import compute_loss
from utils.metrics import compute_accuracy

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
    **kwargs
    ):
    
    """
    Validates the model on the given data_loader and logs the validation loss to W&B.

    Args:
        model: The PyTorch model to validate.
        data_loader: DataLoader providing (ground_truth_coords, input_data) batches.
        loss_choice: Loss function choice ('L1' or 'L2').
        device: Device to run the validation on ('cpu' or 'cuda').

    Returns:
        val_avg_loss: Average validation loss.
        val_avg_acc: Average validation accuracy for each threshold.
    """
    model.eval()
    val_loss = 0.0
    accuracy = []
    
    with torch.no_grad():
        for data in data_loader:
            description = data['description'].to(device)
            gt_target = data['target'].to(device)
            query = data['query'].to(device)
            feature_map = data['feature_map'].to(device)

            # Forward pass: Get predictions
            pred_target = model(description=description, map_tensor=feature_map, query=query)

            # Compute loss
            loss = compute_loss(gt_target, pred_target, loss_choice)
            val_loss += loss.item()
            
            # Compute accuracy
            accuracy.append(compute_accuracy(gt_target, pred_target))
    
    # Calculate average validation loss
    val_avg_loss = val_loss / len(data_loader)
    
    # accuracy = [ {'5': 0.8, '10': 0.9, '20': 1.0}, {'5': 0.8, '10': 0.9, '20': 1.0}, ...]
    val_avg_acc = {key: sum(d[key] for d in accuracy) / len(accuracy) for key in accuracy[0]}
    
    if kwargs.get('mode') in ['eval']:
        log_epoch_metrics(val_avg_loss, val_avg_acc)

    return val_avg_loss, val_avg_acc