# Importing necessary libraries
import numpy as np

# Importing training and validation functions
from trainer.train import train
from trainer.validate import validate

# Importing utility functions
from utils.utils import get_optimizer, get_scheduler, set_seed

# Importing argument parsing function
from args import get_args

# Importing custom models
from model.encoder import Blip2Encoder
from model.model import RetrievalMapModel  

# Dataloader
from dataset.dataloader import get_dataloader

def main():
    
    # Get args and set seed
    args = get_args()
    set_seed(args.seed)
    
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

    # Model
    model = RetrievalMapModel(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        encoder=encoder,
        cosine_method=args.cosine_method,
        pixels_per_meter=args.pixels_per_meter
    )

    # Optimizer
    optimizer = get_optimizer(
        optimizer_name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = get_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        step_size=args.step_size,
        gamma=args.gamma
    )

    if args.mode in ['train']:    
        # Train the model
        train(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            loss_choice=args.loss_choice,
            device=args.device
        )
    elif args.mode in ['eval']:
        # Evaluate the model
        validate(
            model=model,
            data_loader=data_loader,
            device=args.device
        )

if __name__ == "__main__":
    # Parse arguments
    args = get_args()
    main(args)
