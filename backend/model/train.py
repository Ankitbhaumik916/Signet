"""
Training script for Siamese Signature Authentication Model
Trains the CNN on signature datasets (CEDAR, SigNet, etc.)
"""

import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import model architecture
from siamese_model import SiameseNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks
    Pulls matched pairs (label=1) closer, pushes non-matched pairs (label=0) apart
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Maximum distance for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distance: Euclidean distances between embeddings
            label: Binary label (1 for similar pair, 0 for dissimilar)

        Returns:
            Contrastive loss value
        """
        # Loss for similar pairs (label=1): minimize distance
        loss_similar = label * (distance ** 2) / 2

        # Loss for dissimilar pairs (label=0): penalize if distance < margin
        loss_dissimilar = (1 - label) * (torch.clamp(self.margin - distance, min=0.0) ** 2) / 2

        return loss_similar.mean() + loss_dissimilar.mean()


class SignaturePairDataset(Dataset):
    """
    Dataset for signature pairs used in Siamese training
    Creates pairs of genuine and forged signatures
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        transform=None,
        num_pairs: int = None
    ):
        """
        Args:
            data_dir: Directory containing train/val subdirectories
            split: 'train' or 'val'
            transform: Image transformations to apply
            num_pairs: Maximum number of pairs to generate
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.pairs = []
        self.labels = []

        # Load genuine signatures
        genuine_dir = self.data_dir / 'genuine'
        forged_dir = self.data_dir / 'forged'

        if not genuine_dir.exists() or not forged_dir.exists():
            raise ValueError(f"Required directories not found: {genuine_dir}, {forged_dir}")

        genuine_files = sorted(genuine_dir.glob('*.png')) + sorted(genuine_dir.glob('*.jpg'))
        forged_files = sorted(forged_dir.glob('*.png')) + sorted(forged_dir.glob('*.jpg'))

        logger.info(f"Found {len(genuine_files)} genuine and {len(forged_files)} forged signatures")

        # Create positive pairs (genuine-genuine)
        for i, img1 in enumerate(genuine_files):
            for j, img2 in enumerate(genuine_files[i+1:], i+1):
                self.pairs.append((str(img1), str(img2)))
                self.labels.append(1)

        # Create negative pairs (genuine-forged)
        for img1 in genuine_files[:len(forged_files)]:
            for img2 in forged_files:
                self.pairs.append((str(img1), str(img2)))
                self.labels.append(0)

        # Limit number of pairs if specified
        if num_pairs and len(self.pairs) > num_pairs:
            indices = np.random.choice(len(self.pairs), num_pairs, replace=False)
            self.pairs = [self.pairs[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        logger.info(f"Created {len(self.pairs)} pairs ({split} split)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """Get a pair of images and their label"""
        from PIL import Image

        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        # Load images
        img1 = Image.open(img1_path).convert('L')  # Grayscale
        img2 = Image.open(img2_path).convert('L')

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: tuple = (224, 224)
) -> tuple:
    """
    Create training and validation data loaders

    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        image_size: Size to resize images to

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(15),  # Augmentation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = SignaturePairDataset(data_dir, split='train', transform=transform)
    val_dataset = SignaturePairDataset(data_dir, split='val', transform=val_transform)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for img1, img2, label in progress_bar:
        # Move to device
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        # Forward pass
        optimizer.zero_grad()
        distance = model(img1, img2)
        loss = criterion(distance, label)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Validate model

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        
        for img1, img2, label in progress_bar:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Forward pass
            distance = model(img1, img2)
            loss = criterion(distance, label)
            total_loss += loss.item()

            # Simple threshold-based accuracy (distance < 0.5 → similar)
            pred = (distance < 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def train(
    data_dir: Path,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    margin: float = 1.0,
    output_weights: Path = None,
    device: str = 'cuda',
    num_workers: int = 0
):
    """
    Main training function

    Args:
        data_dir: Path to data directory
        epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        margin: Margin for contrastive loss
        output_weights: Path to save model weights
        device: Device to train on ('cuda' or 'cpu')
        num_workers: Number of data loading workers
    """
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info(f"Loading data from {data_dir}")
    train_loader, val_loader = get_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Initialize model
    model = SiameseNetwork(input_channels=1).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss and optimizer
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step()

        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            if output_weights:
                output_path = Path(output_weights)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_path)
                logger.info(f"Saved best model to {output_weights}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save final history
    if output_weights:
        history_path = Path(output_weights).parent / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

    logger.info("Training complete!")


def main():
    """Command-line interface for training"""
    parser = argparse.ArgumentParser(
        description='Train Siamese Signature Authentication Model'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Path to data directory (should contain train/ and val/ subdirs)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.0,
        help='Margin for contrastive loss'
    )
    parser.add_argument(
        '--output-weights',
        type=Path,
        default=Path(__file__).parent / 'weights' / 'siamese_model.pt',
        help='Path to save model weights'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to train on'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    # Start training
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        output_weights=args.output_weights,
        device=args.device,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
