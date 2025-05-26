# Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model


## Setup

```bash
conda create -n compa python=3.11 
conda activate compa

bash setup.sh
```

## Training

### RL-ground for shape area and grid position task

```bash
bash src/scripts/run_shape_spatial_rl_ground.sh

```

### RL

```bash
bash src/scripts/run_shape_spatial_rl.sh
```

### SFT

```bash
bash src/scripts/run_shape_spatial_sft.sh
```

## Evaluation


```bash
bash src/scripts/test_grpo_mm_multigpu.sh
```

