"""
Debug helper for image loading issues.

This script provides helper functions to test image loading from URLs
without needing to run through the entire notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Make sure the KMP_DUPLICATE_LIB_OK env var is set
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_image_loading():
    """Test image loading from URLs with the updated image_from_url function."""
    # Import necessary functions
    print("Importing necessary modules...")
    from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
    from cs231n.image_utils import image_from_url
    
    print("Loading a small subset of COCO data...")
    data = load_coco_data(max_train=50)
    
    print("Sampling a minibatch...")
    batch_size = 2
    captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
    
    print("Testing image loading from URLs...")
    for i, url in enumerate(urls):
        print(f"Loading image {i+1}/{len(urls)} from {url}")
        try:
            img = image_from_url(url)
            if img is not None:
                print(f"✓ Successfully loaded image {i+1} (shape: {img.shape})")
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.title(f"Image {i+1}")
                plt.axis('off')
                plt.show()
            else:
                print(f"✗ Failed to load image {i+1} - image_from_url returned None")
        except Exception as e:
            print(f"✗ Error loading image {i+1}: {str(e)}")
    
    print("Test complete!")

if __name__ == "__main__":
    test_image_loading()
