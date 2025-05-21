import os
import torch
from typing import Optional, Tuple

###########################
# Checkpointing functions #
###########################
def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu'
) -> Tuple[int, float]:
    """
    Loads model (and optimizer/scheduler) state from checkpoint_path.
    Returns (start_epoch, best_val_loss).

    Args:
        model:           your nn.Module
        checkpoint_path: path to .pth checkpoint
        optimizer:       optimizer to restore state_dict into (optional)
        scheduler:       scheduler to restore state_dict into (optional)
        device:          'cpu' or 'cuda'
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start_epoch   = ckpt.get('epoch', 0) + 1
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    print(f"→ Loaded checkpoint '{checkpoint_path}' (epoch {ckpt.get('epoch',0)}); "
          f"resuming at epoch {start_epoch}.")
    return start_epoch, best_val_loss

def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_loss: float
) -> None:
    """
    Saves model + optimizer + scheduler + bookkeeping to checkpoint_path.

    Args:
        checkpoint_path:   file path to write .pth
        model:             your nn.Module
        optimizer:         used optimizer
        scheduler:         lr scheduler (or None)
        epoch:             current epoch (int)
        best_val_loss:     best validation loss so far (float)
    """
    ckpt = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(ckpt, checkpoint_path)
    print(f"← Saved checkpoint to '{checkpoint_path}' (epoch {epoch}, best_val_loss {best_val_loss:.4f})")
