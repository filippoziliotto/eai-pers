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
    print("Blip2Encoder cannot be imported, check your salesforce-lavis dependencies!!!")
from models.model import RetrievalMapModel  
from models.baseline import BaselineModel

def main(args):
    
    # Print all the args
    print("Starting run...")
    
    # Log args set seed and config
    args_logger(args)
    set_seed(args.seed)
    cfg = load_config(config_path=args.config)
    
    # Initialize W&B
    if args.use_wandb:
        wandb.init(project="EAI-Pers", name=cfg.logging.wandb.run_name, config=flatten_config(cfg))
    
    # Get Freezed text encoder and initialize
    encoder = Blip2Encoder(device=args.device, freeze_encoder=args.freeze_encoder)
    encoder.initialize()
        
    # Create the initial Dataset and DataLoader
    #kwargs = vars(args)
    train_loader, val_loader = get_dataloader(
        data_dir=args.data_dir,
        split_dir=args.data_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        augmentation=cfg.augmentations,
    )

    # Model Initialization & Baseline Initialization
    if cfg.baseline.use_baseline:
        model = BaselineModel(
            encoder=encoder,
            type=cfg.baseline,
            device=args.device,
        )
    else:
        model = RetrievalMapModel(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            encoder=encoder,
            type=cfg.model.process_type,
            tau=cfg.model.tau,
            device=args.device,
        )
    print("N° of Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer (and scheduler) initialization using **kwargs for scheduler parameters.
    optimizer, scheduler = get_optimizer(
        optimizer_name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        num_epochs=args.num_epochs,  # for cosine_annealing
        step_size=args.step_size,    # for step_lr
        gamma=args.gamma,            # for any scheduler that uses gamma
    )
    
    # Train and/or validate the model
    if args.mode in ['train']:    
        train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            loss_choice=args.loss_choice,
            device=args.device,
            use_wandb=args.use_wandb,
            mode=args.mode,
            load_checkpoint=args.load_checkpoint,
            save_checkpoint=args.save_checkpoint,
            checkpoint_path=args.checkpoint_path,
            resume_training=args.resume_training,
            validate_every_n_epocs=args.validate_after_n_epochs,
            config=cfg,
        )
    elif args.mode in ['eval']:
        assert args.load_checkpoint, "Checkpoint path must be provided for evaluation."
        validate(
            model=model,
            data_loader=val_loader,
            loss_choice=args.loss_choice,
            device=args.device,
            use_wandb=args.use_wandb,
            load_checkpoint=args.load_checkpoint,
            checkpoint_path=args.checkpoint_path,
            mode=args.mode,
            config=cfg,
        )
        
    # Finish run
    if args.use_wandb:
        wandb.finish()
        
    # Log completion
    print("Run completed.")

if __name__ == "__main__":
        
    # Parse arguments
    args = get_args()
    main(args)
