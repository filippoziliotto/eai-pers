from dataclasses import dataclass
from typing import List
from omegaconf import OmegaConf, DictConfig

# -------------------------------------------------------------------
# DATA CONFIGURATION
# -------------------------------------------------------------------

@dataclass
class DataConfig:
    data_dir: str = "data"  # Root directory containing the dataset
    data_split: str = "object_unseen"  # Dataset split (e.g., seen/unseen scenarios)


# -------------------------------------------------------------------
# AUGMENTATION CONFIGURATION
# Fine-grained control over each type of augmentation
# -------------------------------------------------------------------

@dataclass
class RotationAugmentationConfig:
    use_rotation: bool = False  # Enable random rotation
    angle_range: int = 90  # Maximum rotation angle in degrees
    prob: float = 0.5  # Probability of applying this augmentation

@dataclass
class FlipAugmentationConfig:
    use_horizontal_flip: bool = False  # Enable horizontal flip
    use_vertical_flip: bool = False  # Enable vertical flip
    prob: float = 0.5  # Probability of applying flipping

@dataclass
class RandomCropAugmentationConfig:
    use_crop: bool = False  # Enable random cropping
    max_crop_fraction: float = 0.3  # Max crop size as a fraction of image size
    prob: float = 0.5  # Probability of applying cropping

@dataclass
class DescriptionAugmentationConfig:
    use_desc: bool = False  # Enable textual/description augmentation
    prob: float = 0.5  # Probability of applying this augmentation

@dataclass
class AugmentationsConfig:
    use_aug: bool = False  # Global toggle for augmentation pipeline
    default_prob: float = 0.5  # Default augmentation probability
    flip: FlipAugmentationConfig = FlipAugmentationConfig()
    crop: RandomCropAugmentationConfig = RandomCropAugmentationConfig()
    rotation: RotationAugmentationConfig = RotationAugmentationConfig()
    desc: DescriptionAugmentationConfig = DescriptionAugmentationConfig()

# -------------------------------------------------------------------
# MODEL CONFIGURATION
# Core model architecture, attention, encoder, and map representation
# -------------------------------------------------------------------

@dataclass
class AttentionConfig:
    embed_dim: int = 512  # Embedding dimension for attention modules
    num_heads: int = 8  # Number of attention heads

@dataclass
class ModelConfig:
    type: str = "base"  # Model variant or architecture key
    tau: float = 0.8  # Temperature for softmax or contrastive objectives

@dataclass
class EncoderConfig:
    freeze: bool = True  # Whether to freeze encoder weights during training

@dataclass
class MapConfig:
    size: int = 50  # Spatial map size (e.g., grid resolution)
    embedding_size: int = 768  # Size of the learned spatial representation


# -------------------------------------------------------------------
# TRAINING CONFIGURATION
# Loss function, epochs, validation, and batch management
# -------------------------------------------------------------------

@dataclass
class LossConfig:
    choice: str = "L2"  # Loss type: L1, L2, etc.
    scaling: float = 0.3  # Scaling factor for the loss term

@dataclass
class TrainingConfig:
    batch_size: int = 4  # Mini-batch size
    num_epochs: int = 10  # Number of training epochs
    validate_after_n_epochs: int = 1  # Frequency of validation (in epochs)
    loss: LossConfig = LossConfig()  # Loss-related configuration


# -------------------------------------------------------------------
# OPTIMIZATION CONFIGURATION
# Optimizer and learning rate scheduling
# -------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    type: str = "adam"  # Optimizer type (e.g., Adam, SGD)
    lr: float = 0.001  # Learning rate
    weight_decay: float = 1e-5  # L2 regularization

@dataclass
class SchedulerConfig:
    type: str = "none"  # Scheduler type (e.g., step, plateau)
    step_size: int = 5  # Step size for StepLR
    gamma: float = 0.1  # Decay factor
    patience: int = 10  # Patience for ReduceLROnPlateau


# -------------------------------------------------------------------
# CHECKPOINTING AND DEVICE MANAGEMENT
# -------------------------------------------------------------------

@dataclass
class CheckpointConfig:
    save: bool = False  # Whether to save model checkpoints
    path: str = "model/checkpoints/model.pth"  # Checkpoint file path
    load: bool = False  # Whether to load from an existing checkpoint

@dataclass
class DeviceConfig:
    type: str = "cpu"  # Device type: 'cpu' or 'cuda'
    num_workers: int = 4  # DataLoader parallel workers


# -------------------------------------------------------------------
# LOGGING & DEBUGGING CONFIGURATION
# -------------------------------------------------------------------

@dataclass
class WandbConfig:
    use_wandb: bool = False  # Enable Weights & Biases logging
    run_name: str = "none"  # Name for the W&B run

@dataclass
class LoggingConfig:
    wandb: WandbConfig = WandbConfig()  # Encapsulated wandb config


# -------------------------------------------------------------------
# BASELINE & VISUALIZATION CONFIGURATION
# Used to compare against simple policies or visualize behaviors
# -------------------------------------------------------------------

@dataclass
class BaselineConfig:
    use_baseline: bool = False  # Use a baseline method
    type: str = "random"  # Baseline type (e.g., random, heuristic)


# -------------------------------------------------------------------
# FULL DEFAULT CONFIGURATION
# Top-level schema that combines all sub-configs
# -------------------------------------------------------------------

@dataclass
class DefaultConfig:
    # Data
    data: DataConfig = DataConfig()

    # Random seed
    seed: int = 2025

    # Augmentations
    augmentations: AugmentationsConfig = AugmentationsConfig()

    # Model and training
    training: TrainingConfig = TrainingConfig()
    attention: AttentionConfig = AttentionConfig()
    model: ModelConfig = ModelConfig()
    encoder: EncoderConfig = EncoderConfig()

    # Spatial mapping
    map: MapConfig = MapConfig()

    # Optimization
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    # Checkpointing
    checkpoint: CheckpointConfig = CheckpointConfig()

    # System config
    device: DeviceConfig = DeviceConfig()

    # Logging
    logging: LoggingConfig = LoggingConfig()

    # Debug and visualization
    debug: bool = False
    visualize: bool = False
    use_obstacle_map: bool = False

    # Baseline
    baseline: BaselineConfig = BaselineConfig()

    # External features
    use_extractor: bool = True
