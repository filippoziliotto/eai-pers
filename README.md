

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

4. Download the data from [link: https://www.dropbox.com/scl/fi/17frggambpp9jts07jgop/data.zip?rlkey=9phx05i856ulfdsk93uor6gne&st=si9mjlf5&dl=0], extract it, and it will create a `data/val/` folder. 

5. Make sure you have the LAVIS repository installed on macOS, or install it via conda on Linux. This allows loading the BLIP2 model also used to extract the feature map.
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

Run evaluation with:
```sh
python main.py --config "zs_eval.yaml" --mode "eval"
```

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

---
### Training mode: how the model works

Training follows a two-stage pipeline that turns a map and text into a spatial prediction:

Run training with:
```sh
python main.py --config "train.yaml" --mode "train"
```

1. **Data inputs**  
   Each batch provides a feature map (`feature_map`), a list of scene descriptions (`summary`), a query string (`query`), and the target coordinates (`target`).

2. **Stage 1: Map–text fusion**  
   The first stage (`MapAttentionModel`) encodes the descriptions with the BLIP2 encoder, applies positional embeddings (optional), and runs cross-attention over the flattened map to inject textual context into the spatial features.

3. **Stage 2: Query matching**  
   The second stage (`PersonalizedFeatureMapper`) encodes the query, computes a similarity map over the fused features (cosine similarity or learnable similarity), and applies a soft-argmax to produce predicted coordinates.

4. **Loss + optimization**  
   The training loop computes a coordinate loss (e.g., L2 or Chebyshev-based) from the predicted coordinates (and optionally the value map), backpropagates, and updates model weights.

To train on the difficulty splits in `data/val`, set `training.mode: "train"` and use the `data.train_*` settings to select levels and the train/val split ratio (see `configs/experiments/train.yaml`).
