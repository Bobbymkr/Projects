"""
Unit tests for configuration utilities.
"""

import pytest
import tempfile
from pathlib import Path
from src.utils.config import load_config, ConfigError


class TestConfig:
    """Test configuration utilities."""
    
    def test_load_config_json(self):
        """Test loading JSON configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write('{"test": "value", "number": 42}')
            tmp_path = tmp.name
        
        try:
            config = load_config(tmp_path)
            assert config["test"] == "value"
            assert config["number"] == 42
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_load_config_missing_file(self):
        """Test loading non-existent configuration."""
        with pytest.raises((FileNotFoundError, ConfigError)):
            load_config("nonexistent_file.json")

