
# Local imports
from typing import Optional, Dict
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Other imports
from utils.losses import compute_loss
from utils.metrics import compute_accuracy
from utils.utils import log_lr_scheduler, log_epoch_metrics
from utils.visualize import visualize

# Trainer imports
from trainer.validate import validate
from trainer.utils import load_checkpoint, save_checkpoint

# Get the normalization constant for the loss
loss_norm = None

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_choice: str = 'L2',
    use_wandb: bool = False,
    config: Optional[Dict] = None,
    device: str = 'cpu',
):
    """
    Trains the model for one epoch, logs the training loss and accuracy, and 
    optionally logs metrics to W&B.

    Args:
        model: The PyTorch model to train.
        data_loader: DataLoader providing batches with keys 'description', 'query', 'target', and 'feature_map'.
        optimizer: Optimizer for updating model weights.
        loss_choice: Loss function choice ('L1' or 'L2').
        use_wandb: If True, log metrics to W&B.
        device: Device to run the training on ('cpu' or 'cuda').

    Returns:
        epoch_avg_loss: Average training loss for the epoch (normalized so that the first loss is 1).
        train_avg_acc: Average training accuracy for each threshold for the epoch.
    """
    global loss_norm
    
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    metrics = []

    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Batch", leave=False):
        description, query = data['summary'], data['query']
        gt_target, feature_map = data['target'], data['feature_map']
        
        # Move data to the specified device
        # Convert to float32 for Speedup
        gt_target, feature_map = gt_target.to(torch.float32).to(device), feature_map.to(torch.float32).to(device)
        
        # Forward pass
        output = model(description=description, map_tensor=feature_map, query=query)  # e.g., output dict contains 'value_map'

        # Compute loss and perform backpropagation
        loss = compute_loss(gt_target, output, loss_choice)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Append raw losses
        train_loss = loss.item()
        epoch_loss += train_loss
        num_batches += 1
        
        # Compute accuracy for the batch
        metrics.append(compute_accuracy(gt_target, output))
        
        # Optionally log batch loss to W&B
        if use_wandb:
            wandb.log({"Batch Train Loss": train_loss / len(query), "Batch": batch_idx})
            
        # Visualize predictions if enabled
        if config.debugger.visualize:
            for query_, gt_target_, value_map_, map_path_ in zip(query, gt_target, output['value_map'], data['map_path']):
                visualize(
                    query_, 
                    gt_target_, 
                    value_map_, 
                    map_path_, 
                    batch_idx, 
                    name="prediction", 
                    split="train",
                    use_obstacle_map=config.use_obstacle_map,
                    upscale_factor=2.0
                )
        
        if config.debugger.debug and batch_idx == 0:
            break

    # Compute raw mean batch-sum loss for this epoch
    train_loss /= num_batches
    
    # Calculate average accuracy for the epoch.
    train_avg_metric = {key: sum(d[key] for d in metrics) / len(metrics) for key in metrics[0]}
    
    return train_loss, train_avg_metric

def train_and_validate(
    model, 
    train_loader: DataLoader,
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 10,
    loss_choice: str = 'L2',
    device: str = 'cpu',
    use_wandb: bool = False,
    mode: str = 'train',
    load_checkpoint_: bool = False,
    save_checkpoint_: bool = False,
    checkpoint_path: Optional[str] = None,
    resume_training: Optional[bool] = False,
    validate_every_n_epocs: int = 5,
    config: Optional[Dict] = None,
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
        save_checkpoint (bool): If True, save model weights to checkpoint_path after training (when improved).
        checkpoint_path (str): Path to the checkpoint file.
        resume_training (bool): If True, resume training from the last checkpoint.
        validate_every_n_epocs (int): Run validation (and log validation metrics) every n epochs.
        config: Configuration dictionary for additional settings.
    """
    # Move model to device if not already on it
    if device == 'cuda' and not next(model.parameters()).is_cuda:
        model.to(device)

    # Initialize training variablesx
    start_epoch = 0
    best_val_loss = float('inf')
    first_epoch_loss = None
        
    # Optionally load model weights from checkpoint
    if load_checkpoint_:
        start_epoch, best_val_loss = load_checkpoint(
            model,
            checkpoint_path,
            optimizer=optimizer if resume_training else None,
            scheduler=scheduler,
            device=device
        )
    else:
        print("Starting training from scratch...")
    
    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        # Train for one epoch and get training metrics
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_choice, use_wandb, config, device)

        # Store the first epoch's loss as normalization baseline
        if first_epoch_loss is None: first_epoch_loss = train_loss
            
        # Normalize the training loss
        norm_train_loss = train_loss / first_epoch_loss

        # Always print training metrics
        print(f"\nEpoch {epoch}/{num_epochs} - Train Loss: {norm_train_loss:.4f}")
        print("Train Metrics:")
        for key in train_acc:
            print(f"{key}: {train_acc[key]:.4f}")
        print('-' * 20)

        # Determine if validation should be run this epoch
        if (epoch % validate_every_n_epocs == 0) or ((epoch + 1) == num_epochs):
            # Validate the model and get validation metrics
            val_loss, val_acc = validate(model, val_loader, loss_choice, device, mode=mode, use_wandb=use_wandb, config=config)
            
            # Scheduler step (pass validation loss if ReduceLROnPlateau)
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                log_lr_scheduler(optimizer)
            
            # Optionally save the model if validation loss improves
            if save_checkpoint_ and (val_loss < best_val_loss):
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss
                )
                
            # Print epoch summary with both training and validation metrics
            print(f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}")
            print("Val Metrics:")
            for key in val_acc:
                print(f"{key}: {val_acc.get(key, 0):.4f}")
            print('-' * 20)
            
        else:
            # For epochs without validation, update scheduler if it doesn't depend on validation loss
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                log_lr_scheduler(optimizer)
                
        # Log Metrics to W&B
        if use_wandb:
            log_epoch_metrics(epoch, optimizer, norm_train_loss, train_acc, val_loss if val_loss else None, val_acc if val_acc else None)

    print("Training complete...")