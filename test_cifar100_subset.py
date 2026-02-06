#!/usr/bin/env python3
"""
Test script to verify CIFAR-100 subset functionality.
This will check that the subset selection is deterministic and correct.
"""

import sys
sys.path.insert(0, '/home/maherhanut/Documents/Low-Dimensional-Feedback')

from training_utils.data_loader_factory import get_cifar100_loaders, get_deterministic_class_subset

def test_deterministic_subset():
    """Test that subset selection is deterministic."""
    print("Testing deterministic subset selection...")
    
    # Call multiple times - should get same result
    subset1 = get_deterministic_class_subset(100, 10)
    subset2 = get_deterministic_class_subset(100, 10)
    subset3 = get_deterministic_class_subset(100, 20)
    
    assert subset1 == subset2, "Subset selection should be deterministic!"
    assert len(subset1) == 10, f"Expected 10 classes, got {len(subset1)}"
    assert len(subset3) == 20, f"Expected 20 classes, got {len(subset3)}"
    assert len(set(subset1)) == len(subset1), "Classes should be unique"
    
    print(f"✓ Deterministic subset (10 classes): {subset1}")
    print(f"✓ Deterministic subset (20 classes): {subset3}")
    print()

def test_data_loaders():
    """Test that data loaders work with subsets."""
    print("Testing data loaders with subsets...")
    
    # Test with 10 classes
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=64, 
        root='./data', 
        num_workers=2,
        num_subset_classes=10
    )
    
    # Get a batch
    images, labels = next(iter(train_loader))
    
    print(f"✓ Train loader created with {len(train_loader.dataset)} samples")
    print(f"✓ Test loader created with {len(test_loader.dataset)} samples")
    print(f"✓ Batch shape: {images.shape}")
    print(f"✓ Labels in batch: {labels.unique().tolist()}")
    print(f"✓ Label range: [{labels.min()}, {labels.max()}]")
    
    # Verify labels are in correct range
    assert labels.min() >= 0, "Labels should be >= 0"
    assert labels.max() < 10, f"Labels should be < 10 for 10-class subset, got max={labels.max()}"
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_deterministic_subset()
    test_data_loaders()
