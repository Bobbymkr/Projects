#!/usr/bin/env python3
"""
Model Optimization: Compression, Quantization, and ONNX Conversion.

Usage:
    python scripts/optimize_models.py --agent dqn --compression quantization
    python scripts/optimize_models.py --agent all --compression all
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.quantization
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for quantization", file=sys.stderr)

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX not available. Install with: pip install onnx onnxruntime", file=sys.stderr)

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def quantize_pytorch_model(model_path: Path, output_path: Path) -> Dict[str, Any]:
    """Quantize a PyTorch model to INT8."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate (would need calibration data)
        # For now, just prepare
        
        # Convert to quantized
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        # Calculate size reduction
        original_size = model_path.stat().st_size
        quantized_size = output_path.stat().st_size
        
        return {
            "status": "success",
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else 0,
            "size_reduction_percent": (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
        }
    
    except Exception as e:
        return {"error": str(e)}


def convert_to_onnx(model_path: Path, output_path: Path, input_shape: tuple) -> Dict[str, Any]:
    """Convert model to ONNX format."""
    if not TORCH_AVAILABLE or not ONNX_AVAILABLE:
        return {"error": "PyTorch or ONNX not available"}
    
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Test ONNX model
        ort_session = onnxruntime.InferenceSession(str(output_path))
        
        original_size = model_path.stat().st_size
        onnx_size = output_path.stat().st_size
        
        return {
            "status": "success",
            "original_size_mb": original_size / (1024 * 1024),
            "onnx_size_mb": onnx_size / (1024 * 1024),
            "compression_ratio": original_size / onnx_size if onnx_size > 0 else 0,
            "size_reduction_percent": (1 - onnx_size / original_size) * 100 if original_size > 0 else 0
        }
    
    except Exception as e:
        return {"error": str(e)}


def optimize_agent_model(agent_name: str, compression_type: str = "quantization") -> Dict[str, Any]:
    """Optimize a specific agent's model."""
    models_dir = PROJECT_ROOT / "models"
    optimized_dir = PROJECT_ROOT / "models" / "optimized"
    optimized_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model file
    model_files = list(models_dir.glob(f"*{agent_name}*.pth")) + \
                  list(models_dir.glob(f"*{agent_name}*.pt")) + \
                  list(models_dir.glob(f"*{agent_name}*.h5"))
    
    if not model_files:
        return {"error": f"No model file found for {agent_name}"}
    
    model_file = model_files[0]
    results = {}
    
    if compression_type in ["quantization", "all"]:
        if TORCH_AVAILABLE and model_file.suffix in [".pth", ".pt"]:
            output_path = optimized_dir / f"{agent_name}_quantized.pth"
            result = quantize_pytorch_model(model_file, output_path)
            results["quantization"] = result
    
    if compression_type in ["onnx", "all"]:
        if TORCH_AVAILABLE and ONNX_AVAILABLE and model_file.suffix in [".pth", ".pt"]:
            # Determine input shape based on agent
            input_shapes = {
                "dqn": (8,),  # state_dim
                "model_based_rl": (8,),
                "hierarchical_rl": (8,),
                "transformer": (8,),
            }
            input_shape = input_shapes.get(agent_name, (8,))
            
            output_path = optimized_dir / f"{agent_name}.onnx"
            result = convert_to_onnx(model_file, output_path, input_shape)
            results["onnx"] = result
    
    return {
        "agent": agent_name,
        "original_model": str(model_file),
        "optimizations": results
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize models (compression, quantization, ONNX)")
    parser.add_argument("--agent", type=str, required=True, help="Agent name or 'all'")
    parser.add_argument("--compression", type=str, default="quantization", choices=["quantization", "onnx", "all"], help="Compression type")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    agents_to_optimize = []
    if args.agent == "all":
        agents_to_optimize = ["dqn", "model_based_rl", "hierarchical_rl", "transformer"]
    else:
        agents_to_optimize = [args.agent]
    
    all_results = {}
    
    for agent in agents_to_optimize:
        print(f"\n🔧 Optimizing {agent} model...")
        result = optimize_agent_model(agent, args.compression)
        all_results[agent] = result
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Optimization complete")
            for opt_type, opt_result in result.get("optimizations", {}).items():
                if "error" not in opt_result:
                    print(f"    {opt_type}: {opt_result.get('size_reduction_percent', 0):.1f}% size reduction")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Save to default location
        output_file = PROJECT_ROOT / "results" / "optimization" / f"model_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

