# Importing necessary libraries
import numpy as np
import wandb

# Importing training and validation functions
from trainer.train import train_and_validate
from trainer.validate import validate

# Importing utility functions
from utils.utils import get_optimizer, set_seed, args_logger
from dataset.utils import split_dataloader

# Importing argument parsing function
from args import get_args

# Dataloader
from dataset.dataloader import get_dataloader


# Useful to check if the functions run individually
DEBUG = True

# Importing custom models
if not DEBUG:
    try:
        from models.encoder import Blip2Encoder
    except ImportError:
        print("Blip2Encoder not imported correctly, check your lavis-dependencies.")
from models.model import RetrievalMapModel  


def main(args):
    
    # Print all the args
    print("Starting run...")
    
    # Log args and set seed
    args_logger(args)
    set_seed(args.seed)

    # Initialize W&B
    if args.use_wandb:
        wandb.init(project="EAI-Pers", name=args.run_name)
        
    # Get Freezed text encoder and initialize
    if not DEBUG:
        encoder = Blip2Encoder(device=args.device, freeze_encoder=args.freeze_encoder)
        encoder.initialize()
    else:
        encoder = None

    # Dataset and DataLoader
    kwargs = vars(args)
    data_loader = get_dataloader(
        data_dir=args.data_dir,
        data_split=args.data_split,
        batch_size=args.batch_size,
        shuffle=False,     
        kwargs=kwargs
    )
    
    # Get the different splits from the data_loader
    train_loader, val_loader = split_dataloader(data_loader, split_ratio=0.8, batch_size=args.batch_size)

    # Model Initialization
    model = RetrievalMapModel(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        encoder=encoder,
        cosine_method=args.cosine_method,
        pixels_per_meter=args.pixels_per_meter,
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
            use_wandb=args.use_wandb
            **kwargs
        )
    elif args.mode in ['eval']:
        validate(
            model=model,
            data_loader=val_loader,
            device=args.device,
            loss_choice=args.loss_choice,
            use_wandb=args.use_wandb,
            **kwargs
        )
        
    # Finish run
    if args.use_wandb:
        wandb.finish()
    print("Run completed.")

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    main(args)
