"""
Model Inference Optimization Utilities.

Implements Phase 5.2 from SCORE_IMPROVEMENT_ROADMAP.md:
- Model quantization (INT8)
- Model pruning (50-80% sparsity)
- ONNX export functionality
- Inference benchmarking
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyTorch for optimization
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Model optimization will be limited.")

# Try to import ONNX
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. ONNX export will be disabled.")


class ModelQuantizer:
    """Model quantization utilities for INT8 inference."""
    
    @staticmethod
    def quantize_model_int8(model: Any, calibration_data: list) -> Any:
        """
        Quantize model to INT8 for faster inference.
        
        Args:
            model: PyTorch model to quantize
            calibration_data: Calibration dataset for quantization
            
        Returns:
            Quantized model
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for quantization")
            return model
        
        try:
            # Dynamic quantization (simpler, no calibration needed)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},  # Layers to quantize
                dtype=torch.qint8
            )
            logger.info("Model quantized to INT8")
            return quantized_model
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    @staticmethod
    def get_model_size(model: Any) -> Dict[str, float]:
        """
        Get model size in MB.
        
        Args:
            model: Model to measure
            
        Returns:
            Dictionary with size information
        """
        if not TORCH_AVAILABLE:
            return {"size_mb": 0.0, "params": 0}
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate size (assuming float32 = 4 bytes)
            size_bytes = total_params * 4
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                "size_mb": size_mb,
                "params": total_params,
                "trainable_params": trainable_params,
            }
        except Exception as e:
            logger.warning(f"Failed to get model size: {e}")
            return {"size_mb": 0.0, "params": 0}


class ModelPruner:
    """Model pruning utilities for sparsity."""
    
    @staticmethod
    def prune_model_unstructured(
        model: Any,
        amount: float = 0.5,
        method: str = "magnitude"
    ) -> Any:
        """
        Prune model weights to achieve sparsity.
        
        Args:
            model: PyTorch model to prune
            amount: Fraction of weights to prune (0.0 to 1.0)
            method: Pruning method ("magnitude", "random")
            
        Returns:
            Pruned model
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for pruning")
            return model
        
        try:
            import torch.nn.utils.prune as prune
            
            # Prune all linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if method == "magnitude":
                        prune.l1_unstructured(module, name="weight", amount=amount)
                    elif method == "random":
                        prune.random_unstructured(module, name="weight", amount=amount)
            
            logger.info(f"Model pruned with {amount*100:.1f}% sparsity using {method} method")
            return model
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return model
    
    @staticmethod
    def get_sparsity(model: Any) -> float:
        """
        Calculate model sparsity.
        
        Args:
            model: Model to analyze
            
        Returns:
            Sparsity ratio (0.0 to 1.0)
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        try:
            total_params = 0
            zero_params = 0
            
            for param in model.parameters():
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
            
            if total_params == 0:
                return 0.0
            
            return zero_params / total_params
        except Exception as e:
            logger.warning(f"Failed to calculate sparsity: {e}")
            return 0.0


class ONNXExporter:
    """ONNX model export utilities."""
    
    @staticmethod
    def export_to_onnx(
        model: Any,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11
    ) -> bool:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (batch_size, ...)
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            True if successful, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available for ONNX export")
            return False
        
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available for export")
            return False
        
        try:
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    @staticmethod
    def load_onnx_model(onnx_path: str) -> Optional[Any]:
        """
        Load ONNX model for inference.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            ONNX Runtime session or None
        """
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not available")
            return None
        
        try:
            session = ort.InferenceSession(onnx_path)
            logger.info(f"ONNX model loaded from {onnx_path}")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return None


class InferenceBenchmark:
    """Benchmark inference performance."""
    
    @staticmethod
    def benchmark_inference(
        model: Any,
        input_shape: Tuple[int, ...],
        num_runs: int = 1000,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results dictionary
        """
        if not TORCH_AVAILABLE:
            return {"avg_time_ms": 0.0, "throughput": 0.0}
        
        try:
            import time
            
            model.eval()
            dummy_input = torch.randn(input_shape)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = model(dummy_input)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
            
            avg_time_ms = np.mean(times)
            std_time_ms = np.std(times)
            throughput = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
            
            return {
                "avg_time_ms": avg_time_ms,
                "std_time_ms": std_time_ms,
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "throughput": throughput,  # inferences per second
            }
        except Exception as e:
            logger.warning(f"Benchmarking failed: {e}")
            return {"avg_time_ms": 0.0, "throughput": 0.0}


def optimize_model_for_inference(
    model: Any,
    input_shape: Tuple[int, ...],
    quantization: bool = True,
    pruning: bool = True,
    pruning_amount: float = 0.5,
    export_onnx: bool = False,
    onnx_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive model optimization pipeline.
    
    Args:
        model: Model to optimize
        input_shape: Input tensor shape
        quantization: Whether to quantize
        pruning: Whether to prune
        pruning_amount: Pruning sparsity (0.0 to 1.0)
        export_onnx: Whether to export to ONNX
        onnx_path: Path for ONNX export
        
    Returns:
        Optimization results dictionary
    """
    results = {
        "original_size": ModelQuantizer.get_model_size(model),
        "optimizations_applied": [],
    }
    
    # Pruning
    if pruning:
        model = ModelPruner.prune_model_unstructured(model, amount=pruning_amount)
        results["optimizations_applied"].append(f"pruning_{pruning_amount*100:.0f}%")
        results["sparsity"] = ModelPruner.get_sparsity(model)
        results["pruned_size"] = ModelQuantizer.get_model_size(model)
    
    # Quantization
    if quantization:
        calibration_data = [torch.randn(input_shape) for _ in range(10)]
        model = ModelQuantizer.quantize_model_int8(model, calibration_data)
        results["optimizations_applied"].append("int8_quantization")
        results["quantized_size"] = ModelQuantizer.get_model_size(model)
    
    # ONNX Export
    if export_onnx and onnx_path:
        success = ONNXExporter.export_to_onnx(model, input_shape, onnx_path)
        results["onnx_export"] = success
        if success:
            results["optimizations_applied"].append("onnx_export")
    
    # Benchmark
    results["benchmark"] = InferenceBenchmark.benchmark_inference(
        model, input_shape, num_runs=100
    )
    
    results["final_size"] = ModelQuantizer.get_model_size(model)
    
    return results

