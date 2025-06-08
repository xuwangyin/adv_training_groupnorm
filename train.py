"""
Training script for ResNet on CIFAR-10 with multi-GPU support
Supports both GroupNorm and BatchNorm variants
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os
import sys
from torch.utils.data import DataLoader

# Import model definitions
from resnet_groupnorm import resnet50_groupnorm, resnet101_groupnorm
from resnet_batchnorm import resnet50_batchnorm, resnet101_batchnorm

# Import WideResNet from the specified path
sys.path.append('../robust-ebms-2/rebm/models')
from wide_resnet_innoutrobustness import WideResNet34x10

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet Training')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet101', 'wideresnet34x10'],
                       help='Model architecture (default: resnet50)')
    parser.add_argument('--norm', type=str, default='groupnorm',
                       choices=['groupnorm', 'batchnorm'],
                       help='Normalization type (default: groupnorm)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs to train (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    
    # Multi-GPU and system
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                       help='GPU IDs to use (comma-separated, e.g., "0,1,2,3")')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save best model (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory to store CIFAR-10 data (default: ./data)')
    
    return parser.parse_args()

def get_model(model_name, norm_type, num_classes=10):
    """Get model based on architecture and normalization type"""
    if model_name == 'resnet50':
        if norm_type == 'groupnorm':
            return resnet50_groupnorm(num_classes=num_classes)
        else:
            return resnet50_batchnorm(num_classes=num_classes)
    elif model_name == 'resnet101':
        if norm_type == 'groupnorm':
            return resnet101_groupnorm(num_classes=num_classes)
        else:
            return resnet101_batchnorm(num_classes=num_classes)
    elif model_name == 'wideresnet34x10':
        # WideResNet34x10 with normalization choice
        return WideResNet34x10(
            num_classes=num_classes, 
            activation='relu', 
            dropRate=0.0, 
            return_feature_map=False, 
            normalize_input=False,  # We handle normalization in transforms
            use_groupnorm=True, 
            num_groups=32
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_data_loaders(args):
    """Get CIFAR-10 data loaders with proper augmentation"""
    # CIFAR-10 normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # No augmentation for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)] '
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, test_loader, criterion, device):
    """Validate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def save_checkpoint(state, is_best, save_dir, filename='best_model.pth'):
    """Save best model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        filepath = os.path.join(save_dir, filename)
        torch.save(state, filepath)
        print(f"Best model saved: {filepath}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0)
    
    print(f"Loaded checkpoint from epoch {start_epoch} with best accuracy {best_acc:.2f}%")
    return start_epoch, best_acc

def main():
    args = get_args()
    
    # Set up GPU devices
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Using GPUs: {gpu_ids}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = get_model(args.model, args.norm, num_classes=10)
    model = model.to(device)
    
    # Multi-GPU setup
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                         momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1  # Start from next epoch
    
    print(f"\nStarting training...")
    print(f"Model: {args.model} with {args.norm}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print("-" * 80)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{args.epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            print(f"New best accuracy: {best_acc:.2f}% - Saving model...")
            
            # Save best model
            state = {
                'epoch': epoch,
                'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': args
            }
            save_checkpoint(state, is_best=True, save_dir=args.save_dir, filename='best_model.pth')
        
        print("-" * 80)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
