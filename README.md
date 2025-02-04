### Personalized Maps

Personalized user navigation through retrievable memory maps.

##### Library structure

> **Note:** The following is the structure of the library:\
├── args.py \
├── data/ \
│ ├── train/ \
│ └── val/ \
├── dataset/ \
│ ├── init.py \
│ ├── dataloader.py \
│ ├── load_maps.py \
│ ├── maps/ \
│ │ └── base_map.py \
│ ├── transform.py \
│ └── utils.py \
├── LAVIS/ \
│ ├── ... \
├── main.py \
├── model/ \
│ ├── README.md \
│ ├── encoder.py \
│ ├── extractor.py \
│ ├── model.py \
│ └── stages/ \
│ ├── first_stage.py \
│ └── second_stage.py \
├── requirements.txt \
├── trainer/ \
│ ├── train.py \
│ └── validate.py \
├── utils/ \
│ ├── attention.py \
│ ├── losses.py \
│ ├── similarity.py \
│ └── utils.py \