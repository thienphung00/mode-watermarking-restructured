# Mode Watermarking

A comprehensive watermarking system for Stable Diffusion models that enables invisible, robust watermark embedding and detection for AI-generated images.

## Overview

Mode Watermarking provides a complete solution for watermarking Stable Diffusion generated images. The system embeds imperceptible watermarks during the diffusion process using a novel bias injection strategy, enabling reliable detection while maintaining image quality.

### Key Features

- **Invisible Watermarking**: Non-distortionary embedding that preserves image quality
- **Robust Detection**: Multiple detection methods including UNet-based and Bayesian detectors
- **Per-Sample Determinism**: Unique watermarks per image using deterministic key derivation
- **Survival Analysis**: Tools to measure watermark persistence through denoising steps
- **Type-Safe Configuration**: Pydantic-based configuration with validation
- **Production Ready**: Comprehensive testing, evaluation metrics, and quality assessment

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/thienphung00/mode-watermarking.git
cd mode-watermarking

# Install dependencies
pip install -r requirements.txt

# Or use the installation script
bash install.sh
```

### Development Install

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"
```

## Quick Start

### Generate Watermarked Images

```python
from src.core.config import AppConfig
from src.engine.pipeline import create_pipeline, generate_with_watermark
from src.engine.strategy_factory import create_strategy_from_config

# Load configuration
config = AppConfig.from_yaml("configs/defaults.yaml")

# Create pipeline and strategy
pipeline = create_pipeline(config.diffusion, device="cuda")
strategy = create_strategy_from_config(config.watermark, config.diffusion, device="cuda")

# Generate watermarked image
result = generate_with_watermark(
    pipeline=pipeline,
    strategy=strategy,
    prompt="A beautiful landscape with mountains",
    sample_id="sample_001",
    num_inference_steps=50,
)

# Access results
image = result["image"]  # PIL Image
metadata = result["metadata"]  # Watermark metadata including zT_hash
```

### Command Line Usage

```bash
# Generate images
python examples/generate_images_v2.py \
    --prompts "A cat" "A dog" \
    --output-dir outputs/generated

# Train detector
python examples/train_detector_v2.py \
    --batch-size 32 \
    --epochs 50

# Evaluate detector
python examples/evaluate_detector_v2.py \
    --checkpoint outputs/checkpoints/best_model.pt
```

## Architecture

### Watermarking Pipeline

The system uses **Strategy 1: Pre-Scheduler Injection** to embed watermarks:

```
1. UNet predicts noise: eps_t = UNet(x_t, t, embeddings)
2. Watermark hook injects bias: eps'_t = eps_t + α_t * (M ⊙ G_t)
3. Scheduler updates latent: x_{t-1} = scheduler.step(eps'_t, t, x_t)
4. VAE decodes to image
```

Where:
- `G_t`: Watermark field (G-field) for timestep t
- `α_t`: Bias strength (alpha schedule)
- `M`: Spatial mask (optional)
- `⊙`: Element-wise multiplication

### Key Components

- **`src/algorithms/`**: Core watermarking algorithms
  - `g_field.py`: G-field generation from key streams
  - `keys.py`: Deterministic key derivation
  - `masks.py`: Spatial masking for selective embedding
  - `scheduling.py`: Alpha schedule computation based on target SNR

- **`src/engine/`**: Pipeline integration
  - `pipeline.py`: Stable Diffusion pipeline factory
  - `hooks.py`: Watermark injection hooks
  - `sampling_utils.py`: Custom sampling and survival analysis utilities
  - `strategy_factory.py`: Strategy pattern for watermark modes

- **`src/detection/`**: Watermark detection
  - `recover.py`: Latent recovery and watermark detection
  - `survival.py`: Watermark survival analysis through UNet steps
  - `statistics.py`: Statistical detection methods
  - `metrics.py`: Detection and quality metrics

- **`src/models/`**: Detector models
  - `detectors.py`: UNet-based and Bayesian detectors
  - `layers.py`: Custom neural network layers

## Configuration

Configuration is managed via YAML files with Pydantic validation:

```yaml
watermark:
  mode: "watermarked"  # or "unwatermarked"
  algorithm_params:
    bias:
      mode: "non_distortionary"  # or "distortionary"
      target_snr: 0.05
      alpha_bounds: [0.0, 0.08]
    key_settings:
      key_master: "your-secret-key"
      experiment_id: "exp_001"

diffusion:
  model_id: "runwayml/stable-diffusion-v1-5"
  trained_timesteps: 1000
  inference_timesteps: 50
  guidance_scale: 7.5
  use_fp16: true
```

See `configs/` directory for example configurations.

## Advanced Features

### Watermark Survival Analysis

Measure how watermark perturbations survive through UNet denoising steps:

```python
from src.detection.survival import compute_watermark_survival
from src.engine.sampling_utils import get_text_embeddings

# Get text embeddings
embeddings, _ = get_text_embeddings(pipeline, prompt)

# Compute survival factor
survival_factor, jvp, eps_clean, eps_pert = compute_watermark_survival(
    unet=pipeline.unet,
    x_t=latent_t,
    x_t_pert=latent_t_perturbed,
    delta=watermark_delta,
    t=timestep_tensor,
    encoder_hidden_states=embeddings,
    device="cuda",
)

print(f"Survival factor: {survival_factor:.4f}")
```

See `docs/survival_analysis_usage.md` for detailed examples.

### Custom Sampling with Intermediate Latents

Access intermediate latents during generation:

```python
from src.engine.sampling_utils import custom_ddim_sample

result = custom_ddim_sample(
    pipeline=pipeline,
    prompt="A beautiful landscape",
    num_inference_steps=50,
    return_intermediates=True,
    timesteps_to_save=[0, 10, 20, 30, 40, 49],
)

# Access intermediate latents
intermediate_latents = result["intermediate_latents"]
final_latent = result["latents"]
```

### Hook-Based Intermediate Storage

Store latents and deltas during generation:

```python
from src.engine.pipeline import apply_watermark_hook

with apply_watermark_hook(
    pipeline, 
    strategy, 
    store_intermediates=True,
    timesteps_to_store=[0, 10, 20, 30, 40, 49]
):
    result = pipeline(prompt="A cat", num_inference_steps=50)

# Retrieve stored data
hook = strategy.get_hook()
latents = hook.get_intermediate_latents()
deltas = hook.get_intermediate_deltas()
```

## Detection

### Whitened Matched Filter Detection

```python
from src.detection.recover import detect_watermark

# Detect watermark in latent
score = detect_watermark(
    image_latents=latents,
    key_stream_params={
        "key_master": "secret_key",
        "sample_id": "sample_001",
        "zT_hash": "abc123...",
        "base_seed": 42,
        "experiment_id": "exp_001",
        "domain": "spatial",  # or "frequency"
    },
)

is_watermarked = score > threshold  # e.g., threshold = 0.1
```

### Detector Models

Train and use learned detectors:

```python
from src.models.detectors import UNetDetector, BayesianDetector

# UNet-based detector
detector = UNetDetector(
    input_channels=4,
    base_channels=64,
    num_classes=1,
)

# Bayesian detector
detector = BayesianDetector(
    input_shape=(4, 64, 64),
    pooling_kernel=4,
)
```

## Project Structure

```
mode-watermarking/
├── src/
│   ├── algorithms/      # Core watermarking algorithms
│   ├── core/            # Configuration and interfaces
│   ├── data/            # Dataset and data loading
│   ├── detection/       # Detection and recovery
│   ├── engine/          # Pipeline integration
│   └── models/          # Detector models
├── examples/            # Example scripts
├── configs/             # Configuration files
├── docs/                # Documentation
├── tests/               # Test suite
└── scripts/             # Utility scripts
```

## Documentation

- **Survival Analysis**: `docs/survival_analysis_usage.md`
- **Detector Training**: `docs/Detector_Training.md`
- **Security Analysis**: `docs/SECURITY_ANALYSIS.md`
- **Technical Safety**: `docs/TECHNICAL_SAFETY_ANALYSIS.md`
- **Codebase Summary**: `CODEBASE_SUMMARY.md`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/unit/test_gfield.py
```


**Note**: This is an active research project. The API may change in future versions.
