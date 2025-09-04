"""I/O utilities for file operations and data persistence.

This module provides utilities for file operations, data loading/saving,
and serialization/deserialization of various data formats.
"""

import os
import json
import pickle
import logging
import shutil
import tempfile
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Tuple

# Configure module logger
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_file_write(filepath: Union[str, Path], write_func, mode: str = 'w') -> None:
    """Safely write to a file using a temporary file and atomic rename.
    
    Args:
        filepath: Path to the target file
        write_func: Function that takes a file object and writes to it
        mode: File open mode ('w', 'wb', etc.)
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Create a temporary file in the same directory
    fd, temp_path = tempfile.mkstemp(dir=filepath.parent)
    os.close(fd)
    temp_path = Path(temp_path)
    
    try:
        # Write to the temporary file
        with open(temp_path, mode) as f:
            write_func(f)
        
        # Atomic rename
        if os.name == 'nt' and filepath.exists():  # Windows needs special handling
            temp_path_renamed = temp_path.with_suffix('.renamed')
            os.replace(temp_path, temp_path_renamed)
            os.replace(temp_path_renamed, filepath)
        else:
            os.replace(temp_path, filepath)
    except Exception as e:
        # Clean up the temporary file on error
        if temp_path.exists():
            os.unlink(temp_path)
        raise e


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data as JSON.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Path to save the JSON file
        indent: Indentation level for pretty printing
    """
    def write_json(f):
        json.dump(data, f, indent=indent)
    
    safe_file_write(filepath, write_json)
    logger.debug(f"Saved JSON data to {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON data from {filepath}")
    return data


def save_pickle(data: Any, filepath: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """Save data as a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the pickle file
        protocol: Pickle protocol version
    """
    def write_pickle(f):
        pickle.dump(data, f, protocol=protocol)
    
    safe_file_write(filepath, write_pickle, mode='wb')
    logger.debug(f"Saved pickle data to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    logger.debug(f"Loaded pickle data from {filepath}")
    return data


def save_numpy(data: Any, filepath: Union[str, Path]) -> None:
    """Save data as a NumPy .npy file.
    
    Args:
        data: NumPy array to save
        filepath: Path to save the .npy file
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required to save .npy files")
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    np.save(filepath, data)
    logger.debug(f"Saved NumPy data to {filepath}")


def load_numpy(filepath: Union[str, Path]) -> Any:
    """Load data from a NumPy .npy file.
    
    Args:
        filepath: Path to the .npy file
        
    Returns:
        NumPy array
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required to load .npy files")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NumPy file not found: {filepath}")
    
    data = np.load(filepath)
    logger.debug(f"Loaded NumPy data from {filepath}")
    return data


def save_torch_model(model: Any, filepath: Union[str, Path], save_optimizer: bool = False, 
                    optimizer: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save a PyTorch model.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        save_optimizer: Whether to save optimizer state
        optimizer: PyTorch optimizer (required if save_optimizer is True)
        metadata: Additional metadata to save with the model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to save models")
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Prepare the state dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if save_optimizer:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when save_optimizer is True")
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    # Save the model
    torch.save(save_dict, filepath)
    logger.debug(f"Saved PyTorch model to {filepath}")


def load_torch_model(model: Any, filepath: Union[str, Path], 
                   load_optimizer: bool = False, optimizer: Optional[Any] = None) -> Dict[str, Any]:
    """Load a PyTorch model.
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to the saved model
        load_optimizer: Whether to load optimizer state
        optimizer: PyTorch optimizer (required if load_optimizer is True)
        
    Returns:
        Dictionary with metadata if available
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to load models")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load the state dictionary
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if requested
    if load_optimizer:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when load_optimizer is True")
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning(f"Optimizer state not found in {filepath}")
    
    # Return metadata if available
    metadata = checkpoint.get('metadata', {})
    logger.debug(f"Loaded PyTorch model from {filepath}")
    return metadata


def save_csv(data: List[List[Any]], filepath: Union[str, Path], 
            headers: Optional[List[str]] = None, append: bool = False) -> None:
    """Save data as a CSV file.
    
    Args:
        data: List of rows to save
        filepath: Path to save the CSV file
        headers: Optional list of column headers
        append: Whether to append to an existing file
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    mode = 'a' if append else 'w'
    with open(filepath, mode, newline='') as f:
        writer = csv.writer(f)
        
        # Write headers if provided and not appending
        if headers and not (append and filepath.exists() and filepath.stat().st_size > 0):
            writer.writerow(headers)
        
        # Write data rows
        writer.writerows(data)
    
    logger.debug(f"{'Appended to' if append else 'Saved'} CSV data to {filepath}")


def load_csv(filepath: Union[str, Path], has_headers: bool = True, 
            as_dict: bool = False) -> Union[List[List[str]], List[Dict[str, str]]]:
    """Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        has_headers: Whether the CSV file has a header row
        as_dict: Whether to return rows as dictionaries
        
    Returns:
        List of rows (either as lists or dictionaries)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    with open(filepath, 'r', newline='') as f:
        if as_dict and has_headers:
            reader = csv.DictReader(f)
            data = list(reader)
        else:
            reader = csv.reader(f)
            data = list(reader)
            if has_headers:
                headers = data[0]
                data = data[1:]
    
    logger.debug(f"Loaded CSV data from {filepath}")
    return data


def save_yaml(data: Any, filepath: Union[str, Path]) -> None:
    """Save data as YAML.
    
    Args:
        data: Data to save (must be YAML serializable)
        filepath: Path to save the YAML file
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required to save YAML files")
    
    def write_yaml(f):
        yaml.dump(data, f, default_flow_style=False)
    
    safe_file_write(filepath, write_yaml)
    logger.debug(f"Saved YAML data to {filepath}")


def load_yaml(filepath: Union[str, Path]) -> Any:
    """Load data from a YAML file.
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        Loaded data
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required to load YAML files")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    logger.debug(f"Loaded YAML data from {filepath}")
    return data


def save_dataframe(df: Any, filepath: Union[str, Path], format: str = 'csv') -> None:
    """Save a pandas DataFrame.
    
    Args:
        df: Pandas DataFrame to save
        filepath: Path to save the DataFrame
        format: Format to save as ('csv', 'parquet', 'excel', 'pickle')
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("Pandas is required to save DataFrames")
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    format = format.lower()
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.debug(f"Saved DataFrame to {filepath} as {format}")


def load_dataframe(filepath: Union[str, Path], format: Optional[str] = None) -> Any:
    """Load a pandas DataFrame.
    
    Args:
        filepath: Path to the DataFrame file
        format: Format to load from ('csv', 'parquet', 'excel', 'pickle')
                If None, infer from file extension
        
    Returns:
        Pandas DataFrame
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("Pandas is required to load DataFrames")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DataFrame file not found: {filepath}")
    
    # Infer format from file extension if not provided
    if format is None:
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            format = 'csv'
        elif suffix == '.parquet':
            format = 'parquet'
        elif suffix in ('.xls', '.xlsx'):
            format = 'excel'
        elif suffix == '.pkl':
            format = 'pickle'
        else:
            raise ValueError(f"Cannot infer format from file extension: {suffix}")
    
    format = format.lower()
    if format == 'csv':
        df = pd.read_csv(filepath)
    elif format == 'parquet':
        df = pd.read_parquet(filepath)
    elif format == 'excel':
        df = pd.read_excel(filepath)
    elif format == 'pickle':
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.debug(f"Loaded DataFrame from {filepath} as {format}")
    return df


def copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> None:
    """Copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst}")
    
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    logger.debug(f"Copied file from {src} to {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> None:
    """Move a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst}")
    
    ensure_dir(dst.parent)
    shutil.move(src, dst)
    logger.debug(f"Moved file from {src} to {dst}")


def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
    """List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if recursive:
        return list(directory.glob(f"**/{pattern}"))
    else:
        return list(directory.glob(pattern))


def get_file_size(filepath: Union[str, Path], human_readable: bool = False) -> Union[int, str]:
    """Get the size of a file.
    
    Args:
        filepath: Path to the file
        human_readable: Whether to return size in human-readable format
        
    Returns:
        File size in bytes or human-readable string
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    size_bytes = filepath.stat().st_size
    
    if not human_readable:
        return size_bytes
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024