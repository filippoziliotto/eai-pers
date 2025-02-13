

----
<div align="center">

# Personalized Retrievable Maps using offline memory

This is the repo related to "Personalized user navigation through retrievable memory maps".

  <img src="docs/example.gif" alt="Example Image" width="400">
</div>

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

5. Make sure the `data` directory exists, otherwise create it. Here you will store the `episodes` and the stored retrievable `maps`

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
    export OPENAI_API_KEY="your-openai-key"  

Then, run the following command to load the keys into your environment:  
```sh
source scripts/keys.sh
```

---
### Dataset

The episodes used in the experiments can be found in the data folder, as well as the maps used to retrieve objects. \
Maps are heavy so we leave the link to a dropbox inorder to download them. 
`STAY TUNED FOR THIS!!`

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

If you need to adjust settings, just modify the .sh files for training and/or validation.


