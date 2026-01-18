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
from dataset.dataloader import get_dataloader_new, get_dataloaders_new_split

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
from models.model_lora import TrainedLoraModel

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
    scene_encoder = Blip2Encoder(device=cfg.device.type, freeze_encoder=cfg.encoder.freeze, use_lora=cfg.encoder.lora.use_lora)
    scene_encoder.initialize()
    if cfg.encoder.lora.use_lora:
        query_encoder = Blip2Encoder(device=cfg.device.type, freeze_encoder=cfg.encoder.freeze, use_lora=cfg.encoder.lora.use_lora)
        query_encoder.initialize()
    else:
        query_encoder = scene_encoder
        
    # Create the initial Dataset and DataLoader
    train_loader = None
    val_loader = None
    if cfg.training.mode in ["train"]:
        train_levels_cfg = cfg.data.train_levels
        if train_levels_cfg:
            if isinstance(train_levels_cfg, str):
                train_levels = [train_levels_cfg]
            else:
                train_levels = list(train_levels_cfg)
        else:
            train_levels = ["easy", "medium", "hard"]

        train_loader, val_loader = get_dataloaders_new_split(
            levels=train_levels,
            episodes_base_dir=cfg.data.train_base_dir,
            split_dir=cfg.data.train_split_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.device.num_workers,
            collate_fn=custom_collate,
            augmentation=cfg.augmentations,
            val_ratio=cfg.data.train_val_ratio,
            seed=cfg.data.train_split_seed,
        )
    elif cfg.training.mode in ["eval"]:
        eval_levels_cfg = cfg.data.eval_levels
        if eval_levels_cfg:
            if isinstance(eval_levels_cfg, str):
                eval_levels = [eval_levels_cfg]
            else:
                eval_levels = list(eval_levels_cfg)
        else:
            eval_levels = ["easy", "medium", "hard"]
        val_loader = get_dataloader_new(
            difficulty=eval_levels,
            episodes_base_dir=cfg.data.eval_base_dir,
            split_dir=cfg.data.eval_split_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.device.num_workers,
            collate_fn=custom_collate,
            augmentation=None,
            shuffle=False,
        )

    # Model Initialization & Baseline Initialization
    if cfg.baseline.use_baseline:
        model = BaselineModel(
            encoder=scene_encoder,
            type=cfg.baseline.type,
            device=cfg.device.type,
        )
    elif cfg.model.lora_model.use_trained_lora:
        assert cfg.encoder.lora.use_lora, "Lora model requires Lora to be enabled in the encoder configuration."
        assert cfg.baseline.use_baseline is False, "Lora model cannot be used with baseline."
        model = TrainedLoraModel(
            scene_encoder=scene_encoder,
            query_encoder=query_encoder,
            top_k=cfg.model.lora_model.top_k,
            neighborhood=cfg.model.lora_model.neighborhood,
            nms_radius=cfg.model.lora_model.nms_radius,
            device=cfg.device.type,
        )
    else:
        assert not cfg.model.lora_model.use_trained_lora, "Lora model is not enabled in the configuration."
        model = RetrievalMapModel(
            embed_dim=cfg.attention.embed_dim,
            num_heads=cfg.attention.num_heads,
            ffn_dim=cfg.attention.ffn_dim,
            dropout=cfg.attention.dropout,
            num_cross_layers=cfg.model.fs.num_cross_layers,
            num_self_layers=cfg.model.fs.num_self_layers,
            scene_encoder=scene_encoder,
            query_encoder=query_encoder,
            type=cfg.model.ss.type,
            tau=cfg.model.ss.tau_config,
            use_self_attention=cfg.model.fs.use_self_attention,
            use_pos_embed=cfg.model.fs.use_pos_embed,
            learn_similarity=cfg.model.ss.learn_similarity,
            device=cfg.device.type,
        )
    print("N° of Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Train and/or validate the model
    if cfg.training.mode in ['train']:  
        assert not cfg.baseline.use_baseline, "Baseline cannot be used in training mode."
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
        assert cfg.baseline.use_baseline and not cfg.checkpoint.load, "Evaluation mode only with baseline"
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
