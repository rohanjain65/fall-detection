import os
from os.path import join
from typing import Dict

import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import ImageClassifier
from utils.criterion import Criterion
from utils.misc import send_to_device


def train(
    model: ImageClassifier,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    train_data: DataLoader,
    val_data: DataLoader,
    epochs: int,
    device: str = "cpu",
    output_dir: str = "weights/",
    run_name: str = "cnn",
) -> None:
    """
    Train a fall detection model.

    Args:
        model (FallDetectionModel): The model to train.
        optimizer (optim.Optimizer): The optimizer.
        criterion (Criterion): The loss function.
        train_data (DataLoader): The training data.
        val_data (DataLoader): The validation data.
        epochs (int): The number of epochs to train for.
        device (str, optional): The device to use.
        output_dir (str, optional): The parent directory to save the weights to.
        run_name (str, optional): The name of the run.
    """

    output_dir = join(output_dir, run_name)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    best_metric = float("-inf")

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, criterion, train_data, epoch, device)

        val_losses, val_metrics = evaluate(model, criterion, val_data, epoch, device)

        # Log the validation losses
        wandb.log({"val": {"loss": val_losses, "metric": val_metrics}}, step=wandb.run.step)

        # Save the latest and best model weights
        torch.save(model.state_dict(), join(output_dir, "last.pt"))

        if val_metrics["accuracy"] > best_metric:
            best_metric = val_metrics["accuracy"]
            torch.save(model.state_dict(), join(output_dir, "best.pt"))


def train_one_epoch(
    model: ImageClassifier,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    data: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> None:
    """
    Train a fall detection model for one epoch.

    Args:
        model (FallDetectionModel): The model to train.
        optimizer (optim.Optimizer): The optimizer.
        criterion (Criterion): The loss function.
        data (DataLoader): The training data.
        epoch (int): The current epoch.
        device (str, optional): The device to train on.
    """

    # Set the model to training mode
    model.train()

    for images, targets in tqdm(data, desc=f"Training (Epoch {epoch})", dynamic_ncols=True):
        # Zero the gradients
        optimizer.zero_grad()

        # Send the batch to the training device
        images, targets = send_to_device(images, device), send_to_device(targets, device)

        # Forward pass
        predictions = model(images)

        # Compute the loss
        losses = criterion(predictions, targets)

        # Backward pass
        loss = losses["overall"]
        loss.backward()
        optimizer.step()

        # Log the training losses
        wandb.log({"train": {"loss": losses}}, step=wandb.run.step + len(images))


# TODO: Move metrics to utils/metrics.py
@torch.no_grad()
def evaluate(
    model: ImageClassifier,
    criterion: Criterion,
    data: DataLoader,
    epoch: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate a recommender model.

    Args:
        model (FallDetectionModel): The model to evaluate.
        criterion (Criterion): The loss function.
        data (DataLoader): The data to evaluate.
        epoch (int): The current epoch.
        device (str, optional): The device to evaluate on.

    Returns:
        losses (Dict[str, float]): The average losses.
        metrics (Dict[str, float]): The average metrics.
    """

    # Set the model to evaluation mode
    model.eval()

    # Keep track of the running loss
    losses = {}

    all_targets = []
    all_predictions = []

    for images, targets in tqdm(data, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True):
        # Send the batch to the evaluation device
        images, targets = send_to_device(images, device), send_to_device(targets, device)

        # Forward pass
        predictions = model(images)

        # Compute the loss
        batch_losses = criterion(predictions, targets)

        # Update the running loss
        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        # Update the running metrics
        predicted_labels = predictions.argmax(dim=1).cpu().numpy()
        true_labels = targets.cpu().numpy()

        all_predictions.extend(predicted_labels)
        all_targets.extend(true_labels)

    # Calculate the average losses
    losses = {k: v / len(data) for k, v in losses.items()}

    # Calculate the metrics
    precision = precision_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    accuracy = accuracy_score(all_targets, all_predictions)

    metrics = {"precision": precision, "recall": recall, "accuracy": accuracy}

    return losses, metrics
