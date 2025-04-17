import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader

from data.ntu_dataset import NUM_CLASSES, NTUDataset
from data.transforms import get_train_transforms, get_val_transforms
from model.model import FallDetectionModel
from utils.criterion import FallDetectionCriterion
from utils.misc import load_config, set_random_seed
from utils.processor import train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_random_seed(42)

args = load_config("configs/swin.yaml")

# Load the training & validation data

args["transforms"]["modality"] = args["dataset"]["modality"]

# Load the training & validation data
train_dataset = NTUDataset(**args["dataset"], split="train", transformations=get_train_transforms(**args["transforms"]))

val_dataset = NTUDataset(**args["dataset"], split="val", transformations=get_val_transforms(**args["transforms"]))

# Create the data loaders
train_loader = DataLoader(train_dataset, **args["dataloader"], shuffle=True)
val_loader = DataLoader(val_dataset, **args["dataloader"], shuffle=False)

# Create the model
model = FallDetectionModel(**args["model"], num_classes=NUM_CLASSES, num_channels=train_dataset.get_num_channels()).to(DEVICE)

# Create the optimizer
optimizer = optim.AdamW(model.parameters(), **args["optimizer"])

# Create the loss function
class_frequencies = train_dataset.calculate_class_frequencies().to(DEVICE)

criterion = FallDetectionCriterion(class_frequencies=class_frequencies)

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
