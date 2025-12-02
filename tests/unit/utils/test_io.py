"""
Unit tests for I/O utilities.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.utils.io import save_numpy, load_numpy, save_torch_model, load_torch_model


class TestIO:
    """Test I/O utilities."""
    
    def test_save_load_numpy(self):
        """Test saving and loading numpy arrays."""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            data = np.array([1, 2, 3, 4, 5])
            save_numpy(data, tmp_path)
            
            loaded = load_numpy(tmp_path)
            assert np.array_equal(data, loaded)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_save_torch_model(self):
        """Test saving torch model."""
        # Test with mock model
        try:
            import torch
            model = torch.nn.Linear(10, 5)
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                save_torch_model(model, tmp_path)
                assert Path(tmp_path).exists()
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except ImportError:
            # PyTorch not available, skip
            pytest.skip("PyTorch not available")

