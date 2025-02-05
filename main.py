# Importing necessary libraries
import numpy as np
import wandb

# Importing training and validation functions
from trainer.train import train_and_validate
from trainer.validate import validate

# Importing utility functions
from utils.utils import get_optimizer, get_scheduler, set_seed
from dataset.utils import split_dataloader

# Importing argument parsing function
from args import get_args

# Importing custom models
from model.encoder import Blip2Encoder
from model.model import RetrievalMapModel  

# Dataloader
from dataset.dataloader import get_dataloader

def main(args):
    
    # Get args and set seed
    set_seed(args.seed)
    
    # Print all the args
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
        
    # Initialize W&B
    if args.use_wandb:
        wandb.init(project="EAI-Pers", entity=args.entity, name=args.run_name)
        
    # Get Freezed text encoder and initialize
    encoder = Blip2Encoder(device=args.device, freeze_encoder=args.freeze_encoder)
    encoder.inilialize()

    # Dataset and DataLoader
    kwargs = vars(args)
    data_loader = get_dataloader(
        data_dir=args.data_dir,
        data_split=args.data_split,
        batch_size=args.batch_size,
        shuffle=True,     
        **kwargs
    )
    
    # Get the different splits from the data_loader
    train_loader, val_loader = split_dataloader(data_loader, split_ratio=0.8, batch_size=args.batch_size, **kwargs)

    # Model Initialization
    model = RetrievalMapModel(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        encoder=encoder,
        cosine_method=args.cosine_method,
        pixels_per_meter=args.pixels_per_meter,
    )

    # Optimizer Initialization
    optimizer = get_optimizer(
        optimizer_name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler Initialization
    scheduler = get_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        step_size=args.step_size,
        gamma=args.gamma
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
