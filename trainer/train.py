
# Local imports
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Other imports
from utils.losses import compute_loss
from utils.metrics import compute_accuracy
from trainer.validate import validate

# Config file
import config

def train_one_epoch(
    model, 
    data_loader, 
    optimizer, 
    loss_choice='L2',
    device='cpu',
    ):
    """
    Trains the model for one epoch and logs the training loss to W&B.

    Args:
        model: The PyTorch model to train.
        data_loader: DataLoader providing (ground_truth_coords, input_data) batches.
        optimizer: Optimizer for updating model weights.
        loss_choice: Loss function choice ('L1' or 'L2').
        device: Device to run the training on ('cpu' or 'cuda').

    Returns:
        epoch_avg_loss: Average training loss for the epoch.
        epoch_avg_acc: Average training accuracy for each threshold for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    train_acc = []
    
    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Batch", leave=False):
        description = data['description']
        query = data['query']
        gt_target = data['target'].to(device)
        feature_map = data['feature_map'].to(device)

        # Forward pass
        value_map = model(description=description, map_tensor=feature_map, query=query) # Shape: (batch, w, h)
        
        # Compute loss
        loss = compute_loss(gt_target, value_map, loss_choice, device)
        epoch_loss += loss.item()
        
        # Compute accuracy
        train_acc.append(compute_accuracy(gt_target, value_map, loss_choice))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if config.DEBUG and batch_idx == 2:
            break

    # Calculate average loss for the epoch
    epoch_avg_loss = epoch_loss / len(data_loader)
    
    # Calculate average accuracy for the epoch for each th key
    train_avg_acc = {key: sum(d[key] for d in train_acc) / len(train_acc) for key in train_acc[0]}
    
    return epoch_avg_loss, train_avg_acc

def train_and_validate(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler=None, 
    num_epochs=10, 
    loss_choice='L2',
    device='cpu', 
    use_wandb=False, 
    ):
    """
    Full training loop that trains and validates the model for multiple epochs.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for updating model weights.
        scheduler: Learning rate scheduler (optional).
        num_epochs: Number of epochs to train the model.
        loss_choice: Loss function choice ('L1' or 'L2').
        device: Device to run the training on ('cpu' or 'cuda').
    """

    # Move model to device if not already on it
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        model.to(device)
    
    print("Starting training...")

    def log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
        metrics = {
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
        }
        metrics.update({f"Train Acc [{k}]": v for k, v in train_acc.items()})
        metrics.update({f"Val Acc [{k}]": v for k, v in val_acc.items()})
        wandb.log(metrics)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_choice, device)

        # Validate after each epoch
        val_loss, val_acc = validate(model, val_loader, loss_choice, device)

        # Log metrics to W&B if enabled
        if use_wandb:
            log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

        # Scheduler step (if provided)
        if scheduler:
            scheduler.step()

        # Print epoch summary
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # Print accuracy summary for each th key
        for key in train_acc:
            print(f"Train Acc [{key}]: {train_acc[key]:.4f}, Val Acc [{key}]: {val_acc[key]:.4f}")
        print('-' * 20)

    print("Training complete.")
    

# Example usage (replace with actual dataset, model, optimizer, and scheduler)
if __name__ == '__main__':
    model = ...  # Define your model
    train_loader = DataLoader(...)  # Define your training DataLoader
    val_loader = DataLoader(...)  # Define your validation DataLoader
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_and_validate(model, train_loader, val_loader, optimizer, scheduler, num_epochs=20, loss_choice='L2', device='cuda')