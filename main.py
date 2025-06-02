# Importing necessary libraries
import wandb
import os

# Importing training and validation functions
from trainer.train import train_and_validate
from trainer.validate import validate

# Importing utility functions
from utils.utils import get_optimizer, set_seed, args_logger, read_wandb_api_key
from dataset.utils import custom_collate
from configs.config_utils import load_config, flatten_config

# Importing argument parsing function
from args import get_args

# Dataloader
from dataset.dataloader import get_dataloader

# Avoid LAVIS (useless) FutureWarnings ;)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Safely import the wandb API key
os.environ["WANDB_API_KEY"] = read_wandb_api_key("scripts/keys.sh")

# Importing custom models
try:
    from models.encoder import Blip2Encoder
except ImportError:
    raise ImportError("Blip2Encoder cannot be imported, check your salesforce-lavis dependencies!!!")
from models.model import RetrievalMapModel  
from models.baseline import BaselineModel

def main(args):
    
    # Print all the args
    print("Starting run...")
    
    # Log args set seed and config
    cfg = load_config(config_path=args.config)
    set_seed(cfg.seed)
    
    # Initialize W&B
    if cfg.logging.wandb.use_wandb:
        wandb.init(project="EAI-Pers", name=cfg.logging.wandb.run_name, config=flatten_config(cfg))
    
    # Get Freezed text encoder and initialize
    # TODO: add fake encoder for debugging
    encoder = Blip2Encoder(device=cfg.device.type, freeze_encoder=cfg.encoder.freeze, use_lora=cfg.encoder.lora.use_lora)
    encoder.initialize()
        
    # Create the initial Dataset and DataLoader
    train_loader, val_loader = get_dataloader(
        data_dir=cfg.data.data_dir,
        split_dir=cfg.data.data_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.device.num_workers,
        collate_fn=custom_collate,
        augmentation=cfg.augmentations,
    )

    # Model Initialization & Baseline Initialization
    if cfg.baseline.use_baseline:
        model = BaselineModel(
            encoder=encoder,
            type=cfg.baseline.type,
            device=cfg.device.type,
        )
    else:
        model = RetrievalMapModel(
            embed_dim=cfg.attention.embed_dim,
            num_heads=cfg.attention.num_heads,
            ffn_dim=cfg.attention.ffn_dim,
            dropout=cfg.attention.dropout,
            num_cross_layers=cfg.model.num_cross_layers,
            num_self_layers=cfg.model.num_self_layers,
            encoder=encoder,
            type=cfg.model.type,
            tau=cfg.model.tau,
            use_self_attention=cfg.model.use_self_attention,
            use_pos_embed=cfg.model.use_pos_embed,
            device=cfg.device.type,
        )
    print("N° of Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Train and/or validate the model
    if cfg.training.mode in ['train']:  
        
        # Optimizer (and scheduler) initialization using **kwargs for scheduler parameters.
        optimizer, scheduler = get_optimizer(
            optimizer_name=cfg.optimizer.type,
            model=model,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            scheduler_name=cfg.scheduler.type,
            num_epochs=cfg.training.num_epochs,  # for cosine_annealing
            step_size=cfg.scheduler.step_size,    # for step_lr
            gamma=cfg.scheduler.gamma,           # for any scheduler that uses gamma
        )
      
        train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=cfg.training.num_epochs,
            loss_choice=cfg.training.loss.choice,
            device=cfg.device.type,
            use_wandb=cfg.logging.wandb.use_wandb,
            mode=cfg.training.mode,
            load_checkpoint_=cfg.checkpoint.load,
            save_checkpoint_= cfg.checkpoint.save,
            checkpoint_path= cfg.checkpoint.path,
            resume_training= args.resume_training,
            validate_every_n_epocs=cfg.training.validate_after_n_epochs,
            config=cfg,
        )
    
    elif cfg.training.mode in ['eval']:
        assert not cfg.baseline.use_baseline and not cfg.checkpoint.load, "Evaluation mode does not support baseline or loading checkpoints."
        validate(
            model=model,
            data_loader=val_loader,
            loss_choice=cfg.training.loss.choice,
            device=cfg.device.type,
            use_wandb=cfg.logging.wandb.use_wandb,
            load_checkpoint= cfg.checkpoint.load,
            checkpoint_path= cfg.checkpoint.path,
            mode= cfg.training.mode,
            config=cfg,
        )
        
    # Finish run
    if cfg.logging.wandb.use_wandb:
        wandb.finish()
        
    # Log completion
    print("Run completed.")

if __name__ == "__main__":
        
    # Parse arguments
    args = get_args()
    main(args)
