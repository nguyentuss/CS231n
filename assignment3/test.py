import torch
import unittest
import numpy as np
from cs231n.transformer_layers import PatchEmbedding, TransformerEncoderLayer

# filepath: d:\Work\Study\AI\cs231\assignment3\cs231n\test_transformer_layers.py

class TestPatchEmbedding(unittest.TestCase):
    def test_patch_embedding_shape(self):
        """Test if PatchEmbedding produces the correct output shape."""
        N, C, H, W = 2, 3, 16, 16
        PS = 8  # Patch size
        embed_dim = 8
        
        # Initialize layer
        layer = PatchEmbedding(img_size=H, patch_size=PS, embed_dim=embed_dim)
        
        # Input tensor
        x = torch.randn(N, C, H, W)
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        expected_num_patches = (H // PS) ** 2  # Should be 4 in this case
        self.assertEqual(output.shape, (N, expected_num_patches, embed_dim))

    def test_patch_embedding_values(self):
        """Test if PatchEmbedding produces the expected output values."""
        # Create a deterministic test
        torch.manual_seed(231)
        np.random.seed(231)
        
        N, C, H, W = 2, 3, 16, 16
        PS = 8
        embed_dim = 8
        
        # Initialize layer
        layer = PatchEmbedding(img_size=H, patch_size=PS, embed_dim=embed_dim)
        
        # Use the same seed to reproduce the test from notebook
        x = torch.randn(N, C, H, W)
        
        output = layer(x)
        
        # Expected output from notebook
        expected_output = np.asarray([
            [
                [ 0.17340218, -0.6341526 ,  0.00534353, -0.00208938, -0.13053574,
                 -0.0723498 ,  0.570391  , -0.5484832 ],
                [ 0.75826085,  0.6517286 ,  0.9410556 ,  0.22658484,  0.22908838,
                 0.29188606,  1.1627291 ,  0.20553493],
                [ 0.18826072, -1.175315  , -0.45934296,  0.22657648,  0.05163736,
                 -1.2193633 ,  0.6293135 , -0.06907118],
                [ 0.6661972 ,  0.6862014 , -0.77427   , -0.69206136,  0.19539498,
                 -0.06223261, -0.683141  ,  0.30782953]
            ],
            [
                [-0.01754813,  0.01794782, -0.27840802,  0.6838149 ,  0.29822493,
                 -0.3810072 , -0.83345515,  0.3823208 ],
                [ 0.93228734, -0.06306291, -0.02399439, -0.06193065,  0.3256202 ,
                 -0.44798264,  0.36217704, -0.3620905 ],
                [ 0.02364302, -0.20997411, -0.2952826 , -0.02742453,  0.7600244 ,
                 -0.30619755, -0.67136663,  0.36101747],
                [ 0.21072303, -0.5766786 ,  0.51954585,  0.26895863,  0.04818385,
                 -0.01398893,  0.1810453 ,  0.6056638 ]
            ]
        ])
        
        # Compare outputs
        rel_error = np.max(np.abs(output.detach().numpy() - expected_output) / 
                          (np.maximum(1e-8, np.abs(output.detach().numpy()) + np.abs(expected_output))))
        self.assertLess(rel_error, 1e-4, f"Relative error: {rel_error}")

    def test_patch_embedding_step_by_step(self):
        """Test each step of the PatchEmbedding process individually."""
        # Create a simple input
        torch.manual_seed(0)
        N, C, H, W = 1, 3, 4, 4
        PS = 2
        embed_dim = 3
        
        # Create a controlled input
        x = torch.ones(N, C, H, W)
        x[:, 0, :, :] *= 1  # R channel = 1
        x[:, 1, :, :] *= 2  # G channel = 2
        x[:, 2, :, :] *= 3  # B channel = 3
        
        # Initialize layer with a controlled projection weight
        layer = PatchEmbedding(img_size=H, patch_size=PS, embed_dim=embed_dim)
        with torch.no_grad():
            # Set projection weights to identity-like transformation
            layer.proj.weight.fill_(0.1)
            layer.proj.bias.fill_(0)
        
        # Forward pass
        output = layer(x)
        
        # Manual calculation for verification
        # Step 1: Divide input into patches
        patches_manual = x.reshape(N, C, H//PS, PS, W//PS, PS)
        # Step 2: Permute to group patch dimensions
        patches_manual = patches_manual.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Step 3: Reshape to (N, num_patches, patch_dim)
        patches_manual = patches_manual.reshape(N, (H//PS)*(W//PS), C*PS*PS)
        # Step 4: Apply projection
        expected_output = torch.matmul(patches_manual, 0.1 * torch.ones(C*PS*PS, embed_dim))
        
        # Check if the output matches the manual calculation
        self.assertTrue(torch.allclose(output, expected_output, rtol=1e-5))


class TestTransformerEncoderLayer(unittest.TestCase):
    def test_encoder_layer_shape(self):
        """Test if TransformerEncoderLayer maintains the correct shape."""
        N, S, D = 2, 5, 12
        num_heads = 2
        
        # Initialize layer
        layer = TransformerEncoderLayer(D, num_heads)
        
        # Input tensor
        src = torch.randn(N, S, D)
        
        # Forward pass
        output = layer(src)
        
        # Check output shape
        self.assertEqual(output.shape, (N, S, D))

    def test_encoder_layer_mask(self):
        """Test if TransformerEncoderLayer correctly applies the mask."""
        torch.manual_seed(231)
        np.random.seed(231)
        
        N, S, D = 1, 4, 12
        num_heads = 2
        
        # Initialize layer
        layer = TransformerEncoderLayer(D, num_heads)
        
        # Input tensor
        src = torch.randn(N, S, D)
        
        # Create mask (some elements masked)
        mask = torch.randn(S, S) < 0.5
        
        # Forward passes with and without mask
        output_with_mask = layer(src, mask)
        output_without_mask = layer(src)
        
        # Outputs should be different when mask is applied
        self.assertFalse(torch.allclose(output_with_mask, output_without_mask))

    def test_encoder_layer_values(self):
        """Test if TransformerEncoderLayer produces the expected output values."""
        torch.manual_seed(231)
        np.random.seed(231)
        
        N, S, D = 1, 4, 12
        num_heads = 2
        
        # Initialize layer
        layer = TransformerEncoderLayer(D, num_heads, 4*D)
        
        # Input tensor
        x = torch.randn(N, S, D)
        
        # Create mask
        x_mask = torch.randn(S, S) < 0.5
        
        # Forward pass
        output = layer(x, x_mask)
        
        # Expected output from notebook
        expected_output = np.asarray([
            [[-0.43529928, -0.204897, 0.45693663, -1.1355408, 1.8000772,
              0.24467856, 0.8525885, -0.53586316, -1.5606489, -1.207276,
              1.3986266, 0.3266182],
             [0.06928468, 1.1030475, -0.9902548, -0.34333378, -2.1073136,
              1.1960536, 0.16573538, -1.1772276, 1.2644588, -0.27311313,
              0.29650143, 0.7961618],
             [0.28310525, 0.69066685, -1.2264299, 1.0175265, -2.0517688,
             -0.10330413, -0.5355796, -0.2696466, 0.13948536, 2.0408154,
              0.27095756, -0.25582793],
             [-0.58568114, 0.8019579, -0.9128079, -1.6816932, 1.1572194,
              0.39162305, 0.58195484, 0.7043353, -1.27042, -1.1870497,
              0.9784279, 1.0221335]]
        ])
        
        # Compare outputs
        rel_error = np.max(np.abs(output.detach().numpy() - expected_output) / 
                          (np.maximum(1e-8, np.abs(output.detach().numpy()) + np.abs(expected_output))))
        self.assertLess(rel_error, 1e-6, f"Relative error: {rel_error}")


if __name__ == "__main__":
    unittest.main()