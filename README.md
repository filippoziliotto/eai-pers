

----

# Personalized Retrievable Maps using offline memory

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

All of our pre‐processed data lives under `data/v2/`.  Here’s a breakdown of the important subfolders and files:


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

We provide two evaluation protocols:

1. **`object_unseen`**  
   - **Train:** all *scenes* are “seen,” but only a subset of objects  
   - **Val:** test on *other* objects (in the same scenes) that the model has never encountered  
   - **Use case:** measure generalization to new objects in familiar environments

2. **`scene_unseen`**  
   - **Train:** only a subset of *scenes* (maps), but *all* objects appear  
   - **Val:** test on *other* scenes that the model has never encountered  
   - **Use case:** measure generalization to new environments with known objects

Within each `train/` and `val/` folder is an **`episodes.json`** file.

Before you begin, make sure you have a `data/` directory at the root of this repository:

```bash
mkdir -p data
```
---

### Getting started  

Run the following commands in the terminal:  
```sh
chmod +x scripts/train.sh  
chmod +x scripts/test.sh  
```
---

Before running the scripts, make sure to add your **W&B API key** and **OpenAI API key** to `scripts/keys.sh` as follows:  
    export WANDB_API_KEY="your-wandb-key"  
    export OTHER_API_KEY="your-other-key"

Then, run the following command to load the keys into your environment:  
```sh
source scripts/keys.sh
```

----
### Training and Evaluation

- **Training**: Run the following command to start training with the default settings configuration:  
    ```sh
    scripts/train.sh
    ```
  
- **Evaluation**: To evaluate the model, run:  
    ```sh
    scripts/test.sh
    ```

Experiments configuration parameters can be tuned in the ```configs``` folder. There is a default configuration and in the ```experiments``` subfolder you can overwrite the default variables.

If you need to adjust other settings, just modify the .sh files for training and/or validation as well as the args file.