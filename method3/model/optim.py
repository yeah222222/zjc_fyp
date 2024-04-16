from dataclasses import dataclass, field
from typing import Dict, Optional
import torch.optim as optim

@dataclass
class OptimizerConfig:
    type: str = 'run'
    name: str = 'trainmodelcolor'
    lr: float = 1e-3
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 50.0  # 可以设置为 None 来禁用梯度裁剪
    kwargs: Dict = field(default_factory=lambda: dict())

@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-6
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))
    
    
@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())
    
@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,  # 0
        num_training_steps=100_000,
    ))

@dataclass
class RunConfig:
    name: str = 'debug'
    job: str = 'train'
    mixed_precision: str = 'fp16'  # 'no'
    cpu: bool = False
    seed: int = 42
    val_before_training: bool = False
    vis_before_training: bool = False
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    max_steps: int = 100_000
    checkpoint_freq: int = 1_000
    val_freq: int = 5_000
    vis_freq: int = 5_000
    log_step_freq: int = 20
    print_step_freq: int = 100

    # Inference config
    num_inference_steps: int = 1000
    diffusion_scheduler: Optional[str] = 'ddpm'
    num_samples: int = 1
    num_sample_batches: Optional[int] = None
    sample_from_ema: bool = False 
    sample_save_evolutions: bool = True  # temporarily set by default
    
    # Training config
    freeze_feature_model: bool = True

    # Coloring training config
    coloring_training_noise_std: float = 0.0
    coloring_sample_dir: Optional[str] = None
    
def compute_grad_norm(parameters):
    # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2).item()
    total_norm = 0
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm