from dataclasses import dataclass
from typing import List
from omegaconf import OmegaConf, DictConfig

# -------------------------------------------------------------------
# Structured config definitions matching default.yaml
# -------------------------------------------------------------------

@dataclass
class DataConfig:
    data_dir: str = "data"
    data_split: str = "object_unseen"

@dataclass
class AugmentationsConfig:
    use_aug: bool = False
    aug_prob: float = 0.5
    use_horizontal_flip: bool = False
    use_vertical_flip: bool = False
    use_random_crop: bool = False
    use_random_rotate: bool = False
    use_desc_aug: bool = False

@dataclass
class LossConfig:
    choice: str = "L2"
    scaling: float = 0.3

@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 10
    validate_after_n_epochs: int = 1
    loss: LossConfig = LossConfig()

@dataclass
class AttentionConfig:
    embed_dim: int = 512
    num_heads: int = 8

@dataclass
class ModelConfig:
    type: str = "base"
    tau: float = 0.8

@dataclass
class EncoderConfig:
    freeze: bool = True

@dataclass
class MapConfig:
    size: int = 50
    embedding_size: int = 768

@dataclass
class OptimizerConfig:
    type: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-5

@dataclass
class SchedulerConfig:
    type: str = "none"
    step_size: int = 5
    gamma: float = 0.1
    patience: int = 10

@dataclass
class CheckpointConfig:
    save: bool = False
    path: str = "model/checkpoints/model.pth"
    load: bool = False

@dataclass
class DeviceConfig:
    type: str = "cpu"
    num_workers: int = 4

@dataclass
class WandbConfig:
    use_wandb: bool = False
    run_name: str = "none"

@dataclass
class LoggingConfig:
    wandb: WandbConfig = WandbConfig()

@dataclass
class BaselineConfig:
    use_baseline: bool = False
    type: str = "random"

@dataclass
class DefaultConfig:
    # Data
    data: DataConfig = DataConfig()
    # Seed
    seed: int = 2025
    # Augmentations
    augmentations: AugmentationsConfig = AugmentationsConfig()
    # Model & Training
    training: TrainingConfig = TrainingConfig()
    # Attention
    attention: AttentionConfig = AttentionConfig()
    # Model params
    model: ModelConfig = ModelConfig()
    encoder: EncoderConfig = EncoderConfig()
    # Map
    map: MapConfig = MapConfig()
    # Optimizer & Scheduler
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    # Checkpoints
    checkpoint: CheckpointConfig = CheckpointConfig()
    # Device & performance
    device: DeviceConfig = DeviceConfig()
    # Logging
    logging: LoggingConfig = LoggingConfig()
    # Debug / Visualization
    debug: bool = False
    visualize: bool = False
    use_obstacle_map: bool = False
    # Baseline
    baseline: BaselineConfig = BaselineConfig()
    use_extractor: bool = True