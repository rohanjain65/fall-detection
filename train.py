import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader

from model.model import FallDetectionModel
from utils.criterion import FallDetectionCriterion
from utils.data import FallDetectionDataset
from utils.misc import load_config, set_random_seed
from utils.processor import train
from utils.transforms import get_train_transforms, get_val_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(42)

args = load_config("configs/cnn.yaml")

# Load the training & validation data
train_dataset = FallDetectionDataset(split="train", transformations=get_train_transforms(), **args["dataset"])
val_dataset = FallDetectionDataset(split="test", transformations=get_val_transforms(), **args["dataset"])

# Create the data loaders
train_loader = DataLoader(train_dataset, **args["dataloader"], shuffle=True)
val_loader = DataLoader(val_dataset, **args["dataloader"], shuffle=False)

# Create the model
model = FallDetectionModel(**args["model"]).to(DEVICE)

# Create the optimizer
optimizer = optim.AdamW(model.parameters(), **args["optimizer"])

# Create the loss function
criterion = FallDetectionCriterion(frequencies=train_dataset.frequencies.to(DEVICE))

# Initialize logging
wandb.init(
    dir="./logs",
    project="fall-detection",
    name=args["train"]["run_name"],
    config=args,
    tags=args["tags"],
)


train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_data=train_loader,
    val_data=val_loader,
    device=DEVICE,
    **args["train"],
)
