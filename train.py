import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

# The format of the images that we were given
IMAGE_SIZE = 48
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Create the facial expression dataset here
class FacialExpressionDataset(Dataset):
    # Initialise this dataset here the classes are the different emotions,
    # this converts the categories into numerical categories
    def __init__(self, root: Path, classes: list[str], augment: bool = False):
        self.root = root
        self.classes = classes
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.augment = augment
        self.samples = []
        # finds the images and stores them as an ordered pair of image,class
        for class_name in classes:
            class_samples = [
                (image_path, self.class_to_index[class_name])
                for image_path in sorted((root / class_name).iterdir())
                if image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            # need to check if each emotion folder contains at least one image
            # in the correct format, if I wanted to retrain this on a different set
            # of images with different classes
            if not class_samples:
                raise ValueError(
                    f"No supported images found in {root / class_name}"
                )
            self.samples.extend(class_samples)
    # return the numbers of images
    def __len__(self) -> int:
        return len(self.samples)
    # loading an image and then processing it
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)) # make greyscale
            if self.augment and random.random() < 0.5: # this flips the image horizontally with prob 50%, this helps train the model with more general images, this reduces overfitting
                image = ImageOps.mirror(image)
            pixels = np.asarray(image, dtype=np.float32) / 255.0 # converts pixel value from 0 to 1
        return torch.from_numpy(pixels).unsqueeze(0), label # adds the channel dimension

## Create the CNN: 1) feature extraction 2) classification

# Convolution blocks for feature extraction
class EmotionCNN(nn.Module):
    def __init__(self, number_of_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            #block1 - basic features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # normalises for the next block
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # takes max of 2x2 from every non-overlapping region to reduce comp requirement
            nn.Dropout2d(0.15),# random zeroes out entire feature-map channels to discourage model on relying excessivley on small number of feature i.e. reduce overfitting 
            #block2 - more complex facial features
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # create 64 new feature maps
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2), 
            #block3 - high level facial expressions
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25), 
        )
        # make it into a one-dim vector
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # it hurts but need to do this to reduce overfitting
            nn.Linear(256, number_of_classes), # gives the classification centroids for each emotion yay!
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    @torch.inference_mode()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self(images), dim=1)

# Actually taking the data to the gym (training) - one passthrough
def train_epoch(
    model: EmotionCNN,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True) # to prevent unwanted accumilation btwn batches
        loss = loss_function(model(images), labels)
        loss.backward() # backpropagation
        optimizer.step() # update weight
        total_loss += loss.item() * labels.size(0) # get average training loss
    return total_loss / len(loader.dataset)

## Evaluate on validation images
def evaluate(
    model: EmotionCNN,
    loader: DataLoader,
    loss_function: nn.Module,
    number_of_classes: int,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct_per_class = torch.zeros(number_of_classes, dtype=torch.float64)
    count_per_class = torch.zeros(number_of_classes, dtype=torch.float64)
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            for class_index in range(number_of_classes):
                class_mask = labels == class_index
                count_per_class[class_index] += class_mask.sum().item()
                correct_per_class[class_index] += (
                    (predictions == labels) & class_mask
                ).sum().item()
    accuracy = correct_per_class.sum() / count_per_class.sum()
    balanced_accuracy = (correct_per_class / count_per_class).mean()
    return total_loss / len(loader.dataset), accuracy.item(), balanced_accuracy.item()
# this outputs the accuracy and then balanced accuracy because different number of images per emotion


def prompt_training_settings(args: argparse.Namespace) -> None:
    settings = (
        ("epochs", "Epochs", int),
        ("batch_size", "Batch size", int),
        ("learning_rate", "Learning rate", float),
    )
    for attribute, label, value_type in settings:
        default = getattr(args, attribute)
        while True:
            response = input(f"{label} [{default}]: ").strip()
            if not response:
                break
            try:
                value = value_type(response)
            except ValueError:
                print(f"Please enter a valid {label.lower()}.")
                continue
            if value <= 0:
                print(f"{label} must be greater than zero.")
                continue
            setattr(args, attribute, value)
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CNN to classify facial expressions from image folders."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--validation-dir", type=Path, default=Path("data/validation"))
    parser.add_argument("--output", type=Path, default=Path("models/fer_cnn.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive; learning-rate must be > 0")

    if sys.stdin.isatty():
        prompt_training_settings(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.train_dir.is_dir() or not args.validation_dir.is_dir():
        raise FileNotFoundError("Training and validation directories must both exist.")

    classes = sorted(
        directory.name for directory in args.train_dir.iterdir() if directory.is_dir()
    )
    if not classes:
        raise ValueError(f"No class folders found in {args.train_dir}")
    validation_classes = {
        directory.name
        for directory in args.validation_dir.iterdir()
        if directory.is_dir()
    }
    if validation_classes != set(classes):
        raise ValueError(
            "Training and validation folders must contain the same class names. "
            f"Train: {classes}; validation: {sorted(validation_classes)}"
        )

    train_dataset = FacialExpressionDataset(args.train_dir, classes, augment=True)
    validation_dataset = FacialExpressionDataset(
        args.validation_dir, classes, augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(len(classes)).to(device)
    class_counts = torch.bincount(
        torch.tensor([label for _, label in train_dataset.samples]),
        minlength=len(classes),
    ).float()
    class_weights = len(train_dataset) / (len(classes) * class_counts) # to handle class imabalance by normalising
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    print(
        f"Device: {device} | Classes: {', '.join(classes)} | "
        f"Training images: {len(train_dataset)} | "
        f"Validation images: {len(validation_dataset)}"
    )
    best_balanced_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        validation_loss, accuracy, balanced_accuracy = evaluate(
            model, validation_loader, loss_function, len(classes), device
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} - "
            f"loss: {training_loss:.4f} - val_loss: {validation_loss:.4f} - "
            f"val_accuracy: {accuracy:.4f} - "
            f"val_balanced_accuracy: {balanced_accuracy:.4f}"
        )

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                    "best_validation_balanced_accuracy": balanced_accuracy,
                    "epoch": epoch,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
