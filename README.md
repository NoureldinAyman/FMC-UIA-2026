# FMC_UIA 2026: Ultrasound Foundation Model Challenge

Entry for the Foundation Model Challenge for Ultrasound Image Analysis at ISBI 2026. One shared model handles 27 ultrasound tasks at once, instead of one model per task.

## Tasks

| Type | Tasks | Examples |
|---|---|---|
| Segmentation | 12 | Fetal head, cardiac structures, thyroid nodules |
| Classification | 9 | Breast lesions, lung disease, fetal planes |
| Detection | 3 | Thyroid nodules, uterine fibroids |
| Regression | 3 | Anatomical landmarks |

## Model

- **Shared encoder:** EfficientNet-B7
- **Shared decoder:** Feature Pyramid Network with scSE attention
- **Task heads:** a separate head for each task, including Mish segmentation heads, MLP regression heads and a grid detection head

## Training

- **Data:** a dataset of 40,388 images, split 80/20 into training and validation
- **Sampling:** a sampler that keeps tasks evenly represented
- **Augmentation:** elastic transforms and grid distortion, chosen to suit ultrasound
- **Losses:** Dice and focal loss for segmentation and CIoU for detection, balanced across tasks with learned uncertainty weights
- **Optimization:** mixed precision, gradient accumulation and a warmup cosine learning-rate schedule
- **Inference:** four-way flip test-time augmentation

## Repository

- `train.py`: training loop
- `model_factory.py`: shared encoder, decoder and task heads
- `dataset.py`: data loading and task sampling
- `utils.py`: losses and metrics
- `model.py`: inference
- `evaluate.py`, `visualize.py`: evaluation and visualization
- `docker/`: challenge submission image

To run it, install `colab_requirements.txt`, set `DATA_ROOT_PATH` in `train.py` and run `python train.py`. The dataset and trained weights are not stored here. The data comes from the [challenge organizers](https://github.com/lijiake2408/Foundation-Model-Challenge-for-Ultrasound-Image-Analysis), whose baseline code this project starts from. They released that baseline for competition use.
