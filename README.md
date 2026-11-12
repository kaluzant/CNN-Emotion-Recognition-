# Facial expression CNN

This project trains a convolutional neural network on the image-folder dataset in
`data/train` and evaluates it on `data/validation`. Each split must contain one
subfolder per expression class. Images are converted to 48×48 grayscale and
scaled to `[0, 1]`; the class names are read from the training subfolders.

## Setup

Use Python with PyTorch support, then install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 train.py
```

The default run trains for 30 epochs with batches of 64. It automatically uses
CUDA when available, applies class-weighted loss for the imbalanced classes, and
saves the checkpoint with the best validation balanced accuracy to
`models/fer_cnn.pth`.

To adjust the run:

```bash
python3 train.py --epochs 40 --batch-size 32 --learning-rate 0.0005
```

The checkpoint contains the model weights, ordered class names, and 48×48 input
size. The later camera application can use these values to keep preprocessing
and predicted labels consistent with training.

## Inference probabilities

`EmotionCNN.predict_proba()` returns one probability per class for each input
image. The probability columns follow the ordered `classes` list stored in the
checkpoint:

```python
checkpoint = torch.load("models/fer_cnn.pth", map_location="cpu")
model = EmotionCNN(len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model_state_dict"])

probabilities = model.predict_proba(images)  # shape: (batch_size, number_of_classes)
```

Provide `images` as grayscale `[0, 1]` tensors with shape
`(batch_size, 1, 48, 48)`, using the same preprocessing as the training dataset.
Each row contains probabilities that sum to 1.
