
# Local imports
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os

# Other imports
from utils.losses import compute_loss
from utils.metrics import compute_accuracy
from trainer.validate import validate

# Config file
import config
from typing import Optional, Dict

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_choice: str = 'NCE',
    device: str = 'cpu',
    ):
    
    """
    Trains the model for one epoch, logs the training loss and accuracy, and 
    optionally loads and/or saves model weights to a checkpoint file.

    Args:
        model: The PyTorch model to train.
        data_loader: DataLoader providing batches with keys 'description', 'query', 'target', and 'feature_map'.
        optimizer: Optimizer for updating model weights.
        loss_choice: Loss function choice ('L1' or 'L2').
        device: Device to run the training on ('cpu' or 'cuda').

    Returns:
        epoch_avg_loss: Average training loss for the epoch.
        train_avg_acc: Average training accuracy for each threshold for the epoch.
    """
            
    # Set model to training mode
    model.train()
    epoch_loss = 0.0
    train_acc = []
    
    # Iterate over the data loader
    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Batch", leave=False):
        description = data['description']
        query = data['query']
        gt_target = data['target'].to(torch.float32).to(device)
        feature_map = data['feature_map'].to(device)

        # Forward pass
        value_map = model(description=description, map_tensor=feature_map, query=query) # Shape: (batch, w, h)
        
        # Compute loss
        loss, pred_target = compute_loss(gt_target, value_map, loss_choice, device)
        epoch_loss += loss.item()
        
        # Compute accuracy
        train_acc.append(compute_accuracy(gt_target, pred_target))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if config.DEBUG and batch_idx == 1:
            break
        
    # Calculate average loss for the epoch
    epoch_avg_loss = epoch_loss / len(data_loader)
    
    # Calculate average accuracy for the epoch for each th key
    train_avg_acc = {key: sum(d[key] for d in train_acc) / len(train_acc) for key in train_acc[0]}
    
    return epoch_avg_loss, train_avg_acc

def train_and_validate(
    model, 
    train_loader: DataLoader,
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler= Optional[torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int=10,
    loss_choice: str='L2',
    device: str='cpu',
    use_wandb: bool=False,
    mode: str='train',
    load_checkpoint: bool=False,
    save_checkpoint: bool=False,
    checkpoint_path: Optional[str]=None,
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
        use_wandb: If True, log metrics to W&B.
        mode: Mode of operation ('train' or 'eval').
        load_checkpoint (bool): If True, load model weights from checkpoint_path before training.
        save_checkpoint (bool): If True, save model weights to checkpoint_path after training.
        checkpoint_path (str): Path to the checkpoint file.
    """

    # Move model to device if not already on it
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        model.to(device)
        
    # Optionally load model weights from checkpoint
    if load_checkpoint:
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist."
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}...")
    else:
        print(f"Starting training from scratch...")
    
    print("Starting training...")

    # Log metrics to W&B if enabled
    def log_epoch_metrics(epoch, train_loss, train_acc, val_loss, val_acc):
        metrics = {
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
        }
        metrics.update({f"Train Acc [{k}]": v for k, v in train_acc.items()})
        metrics.update({f"Val Acc [{k}]": v for k, v in val_acc.items()})
        wandb.log(metrics)

    # Train for the specified number of epochs
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_choice, device)

        # Validate after each epoch
        val_loss, val_acc = validate(model, val_loader, loss_choice, device, mode='train')

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
        
        # Optionally save model weights to checkpoint
        if save_checkpoint:
            assert os.path.exists(os.path.dirname(checkpoint_path)), f"Checkpoint directory {os.path.dirname(checkpoint_path)} does not exist."
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}...")

    print("Training complete...")
    

# Example usage (replace with actual dataset, model, optimizer, and scheduler)
if __name__ == '__main__':
    model = ...  # Define your model
    train_loader = DataLoader(...)  # Define your training DataLoader
    val_loader = DataLoader(...)  # Define your validation DataLoader
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_and_validate(model, train_loader, val_loader, optimizer, scheduler, ...)