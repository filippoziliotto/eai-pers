

----

# PersONAL: Towards a Comprehensive Benchmark for Personalized Embodied AI Agents

This is the repo related to "Personalized user navigation through retrievable memory maps".

---

### Installation

The **EAI-Pers** package supports **Python 3.9 or higher**.  

---

To install the **EAI-Pers** package, make sure you have Python 3.9 or higher.

1. Create the `conda` environment using

    ```sh
    conda create --name eai-pers python=3.9
    ```

2. Clone the repository:  
    ```sh
    git clone https://github.com/filippoziliotto/eai-pers.git
    cd eai-pers
    ```

3. Install the required packages:  
    ```sh
    pip install -r requirements.txt
    ```

4. Install the package using `setup.py`:  
    ```sh
    python setup.py install
    ```

---

### Dataset Structure

All of our pre‐processed data lives under `data/val/`. Here’s a breakdown of the important subfolders and files:

#### 1. `maps/`

- **`embed_dicts/`**  
  Contains per‐map feature‐map embeddings that your model will read in at runtime.

- **`init_dicts/`**  
  Stores the initial robot and object states for each map.

- **`step_actions/`**  
  Steps action for feature map extraction reproducibility.

- **`id_to_floor.json`**  
  A simple lookup table mapping each map’s unique ID to its floor (e.g., “floor0”, “floor1”).

#### 2. `splits/`

Difficulty splits live under `data/val/splits/{easy,medium,hard}`. Each level contains a `content/` directory with the episode JSON files used to build the dataset.

---
### Zero-shot baseline and evaluation flow

Evaluation uses the difficulty splits from `data/val/splits/{easy,medium,hard}`. Training can use the same dataset by splitting episodes into train/val via the `data.train_*` settings.

#### How to write the evaluation code

Evaluation is wired in `main.py` by switching the data loader based on `training.mode`. For a minimal eval-only setup, use `configs/experiments/zs_eval.yaml`, which contains only the required config keys:

- Set `training.mode: "eval"` in your experiment config (see `configs/experiments/zs_eval.yaml`).
- Use the eval dataset settings from the `data` config:
  - `eval_base_dir: "data/val"`
  - `eval_split_dir: "splits"`
  - `eval_levels: ["easy", "medium", "hard"]` to merge all levels into a single eval dataset.

Minimal evaluation wiring looks like this:

```python
if cfg.training.mode == "eval":
    val_loader = get_dataloader_new(
        difficulty=cfg.data.eval_levels,
        episodes_base_dir=cfg.data.eval_base_dir,
        split_dir=cfg.data.eval_split_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.device.num_workers,
        collate_fn=custom_collate,
        augmentation=None,
        shuffle=False,
    )
```

#### zs_cosine baseline: what it does

`baseline.type: "zs_cosine"` runs a zero-shot cosine matcher:

- Encodes the **query** text into a single embedding.
- Encodes **scene descriptions** into a sequence of embeddings.
- Uses description embeddings to score each spatial location, keeps the top-k locations per description with NMS, and builds a spatial mask.
- Computes cosine similarity between the **query** embedding and the masked **feature map**.
- Returns the location with the maximum similarity as the prediction.
