"""Configuration management utilities.

This module provides utilities for loading, validating, and managing
configuration settings for the adaptive traffic control system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

# Try to import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from omegaconf import OmegaConf, DictConfig
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a file.
    
    Supports JSON and YAML formats based on file extension.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If the file cannot be loaded or parsed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    try:
        if suffix == '.json':
            with open(config_path, 'r') as f:
                return json.load(f)
        elif suffix in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ConfigError("PyYAML is required to load YAML configuration files")
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ConfigError(f"Unsupported configuration file format: {suffix}")
    except Exception:
        raise ConfigError(f"Failed to load configuration from {config_path}.")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.
    
    The override_config values take precedence over base_config values.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, override_value in override_config.items():
        if (
            key in result and 
            isinstance(result[key], dict) and 
            isinstance(override_value, dict)
        ):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], override_value)
        else:
            # Override or add the value
            result[key] = override_value
    
    return result


def load_config_with_overrides(base_path: Union[str, Path], 
                              override_paths: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
    """Load a base configuration with optional overrides.
    
    Args:
        base_path: Path to the base configuration file
        override_paths: List of paths to override configuration files
        
    Returns:
        Merged configuration dictionary
    """
    config = load_config(base_path)
    
    if override_paths:
        for path in override_paths:
            override_config = load_config(path)
            config = merge_configs(config, override_config)
    
    return config


def validate_config_schema(config: Dict[str, Any], schema_class: Any) -> Any:
    """Validate configuration against a Pydantic schema.
    
    Args:
        config: Configuration dictionary to validate
        schema_class: Pydantic model class for validation
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ConfigError: If validation fails
    """
    if not PYDANTIC_AVAILABLE:
        raise ConfigError("Pydantic is required for configuration validation")
    
    try:
        return schema_class(**config)
    except ValidationError as e:
        raise ConfigError(f"Configuration validation failed: {str(e)}") from e


def load_hydra_config(config_path: str, config_name: str) -> Any:
    """Load configuration using Hydra.
    
    Args:
        config_path: Path to the configuration directory
        config_name: Name of the configuration file
        
    Returns:
        Hydra DictConfig object
        
    Raises:
        ConfigError: If Hydra is not available or loading fails
    """
    if not OMEGACONF_AVAILABLE:
        raise ConfigError("Hydra/OmegaConf is required to load Hydra configurations")
    
    try:
        # Import here to avoid dependency if not used
        import hydra
        from hydra.core.config_store import ConfigStore
        
        # Initialize Hydra
        hydra.initialize(config_path=config_path)
        
        # Load the configuration
        cfg = hydra.compose(config_name=config_name)
        return cfg
    except ImportError:
        raise ConfigError("Hydra is required to load Hydra configurations")
    except Exception:
        raise ConfigError("Failed to load Hydra configuration.")


def get_nested_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'database.host')
        default: Default value to return if the key is not found
        
    Returns:
        The configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to the value (e.g., 'database.host')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to a file.
    
    Supports JSON and YAML formats based on file extension.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
        
    Raises:
        ConfigError: If the file cannot be saved
    """
    config_path = Path(config_path)
    
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = config_path.suffix.lower()
    try:
        if suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif suffix in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ConfigError("PyYAML is required to save YAML configuration files")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ConfigError(f"Unsupported configuration file format: {suffix}")
    except Exception:
        raise ConfigError(f"Failed to save configuration to {config_path}.")


def load_environment_config(prefix: str = "APP_") -> Dict[str, Any]:
    """Load configuration from environment variables.
    
    Environment variables with the specified prefix are converted to a nested
    configuration dictionary using double underscore as a separator.
    
    For example, APP_DATABASE__HOST=localhost becomes {'database': {'host': 'localhost'}}
    
    Args:
        prefix: Prefix for environment variables to include
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and split by double underscore
            key_without_prefix = key[len(prefix):]
            parts = key_without_prefix.split('__')
            
            # Convert value to appropriate type
            if value.lower() == 'true':
                typed_value = True
            elif value.lower() == 'false':
                typed_value = False
            elif value.isdigit():
                typed_value = int(value)
            elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                typed_value = float(value)
            else:
                typed_value = value
            
            # Build nested dictionary
            current = config
            for part in parts[:-1]:
                part = part.lower()
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[parts[-1].lower()] = typed_value
    
    return config


def create_config_from_template(template_path: Union[str, Path], 
                               output_path: Union[str, Path],
                               replacements: Dict[str, Any]) -> None:
    """Create a configuration file from a template.
    
    Args:
        template_path: Path to the template file
        output_path: Path to save the output file
        replacements: Dictionary of replacements
        
    Raises:
        ConfigError: If the template cannot be loaded or the output cannot be saved
    """
    template_path = Path(template_path)
    output_path = Path(output_path)
    
    if not template_path.exists():
        raise ConfigError(f"Template file not found: {template_path}")
    
    try:
        # Read the template
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Apply replacements
        output_content = template_content
        for key, value in replacements.items():
            placeholder = f"{{{{ {key} }}}}"
            output_content = output_content.replace(placeholder, str(value))
        
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the output
        with open(output_path, 'w') as f:
            f.write(output_content)
    except Exception:
        raise ConfigError("Failed to create configuration from template.")


def register_hydra_config_schemas(config_schemas: Dict[str, Any]) -> None:
    """Register Pydantic models as Hydra structured configs.
    
    Args:
        config_schemas: Dictionary mapping config names to Pydantic model classes
        
    Raises:
        ConfigError: If Hydra or Pydantic is not available
    """
    if not OMEGACONF_AVAILABLE:
        raise ConfigError("Hydra/OmegaConf is required to register config schemas")
    
    if not PYDANTIC_AVAILABLE:
        raise ConfigError("Pydantic is required to register config schemas")
    
    try:
        from hydra.core.config_store import ConfigStore
        cs = ConfigStore.instance()
        
        for name, schema in config_schemas.items():
            cs.store(name=name, node=schema)
    except ImportError:
        raise ConfigError("Hydra is required to register config schemas")
    except Exception:
        raise ConfigError("Failed to register Hydra config schemas.")


def create_default_configs(config_dir: Union[str, Path], config_templates: Dict[str, Dict[str, Any]]) -> None:
    """Create default configuration files.
    
    Args:
        config_dir: Directory to save configuration files
        config_templates: Dictionary mapping file names to default configurations
        
    Raises:
        ConfigError: If the configuration files cannot be saved
    """
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, config in config_templates.items():
        file_path = config_dir / filename
        
        # Skip if file already exists
        if file_path.exists():
            logger.info(f"Configuration file already exists: {file_path}")
            continue
        
        # Save the configuration
        try:
            save_config(config, file_path)
            logger.info(f"Created default configuration file: {file_path}")
        except Exception:
            logger.error(f"Failed to create default configuration file {file_path}.")


def validate_config_with_function(config: Dict[str, Any], 
                                 validator_func: Callable[[Dict[str, Any]], List[str]]) -> List[str]:
    """Validate configuration using a custom validation function.
    
    Args:
        config: Configuration dictionary to validate
        validator_func: Function that takes a config and returns a list of error messages
        
    Returns:
        List of validation error messages (empty if valid)
        
    Raises:
        ConfigError: If validation fails and raise_error is True
    """
    errors = validator_func(config)
    return errors


def get_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """Get the differences between two configuration dictionaries.
    
    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary
        
    Returns:
        Dictionary of differences
    """
    diff = {}
    
    # Find keys in config2 that differ from config1
    for key, value2 in config2.items():
        if key not in config1:
            diff[key] = {'added': value2}
        elif isinstance(value2, dict) and isinstance(config1[key], dict):
            nested_diff = get_config_diff(config1[key], value2)
            if nested_diff:
                diff[key] = nested_diff
        elif config1[key] != value2:
            diff[key] = {'old': config1[key], 'new': value2}
    
    # Find keys in config1 that are not in config2
    for key, value1 in config1.items():
        if key not in config2:
            diff[key] = {'removed': value1}
    
    return diff


def get_config_schema(schema_class: Any) -> Dict[str, Any]:
    """Get a JSON schema from a Pydantic model class.
    
    Args:
        schema_class: Pydantic model class
        
    Returns:
        JSON schema dictionary
        
    Raises:
        ConfigError: If Pydantic is not available
    """
    if not PYDANTIC_AVAILABLE:
        raise ConfigError("Pydantic is required to get config schema")
    
    try:
        return schema_class.schema()
    except Exception:
        raise ConfigError("Failed to get config schema.")