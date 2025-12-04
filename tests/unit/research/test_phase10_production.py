"""
Comprehensive Unit Tests for Phase 10 Production Readiness.

Tests:
- Mixed Precision Training
- Model Quantization
- Model Pruning
- Knowledge Distillation
- Metrics Collection
- Production Optimizer
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.research.production.phase10_production import (
    MixedPrecisionTrainer,
    ModelQuantizer,
    ModelPruner,
    KnowledgeDistillation,
    MetricsCollector,
    PerformanceMetrics,
    SystemMetrics,
    ModelMetrics,
    ProductionOptimizer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestMixedPrecisionTrainer:
    """Test Mixed Precision Training."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    @pytest.fixture
    def trainer(self, model):
        return MixedPrecisionTrainer(model)
    
    def test_initialization(self, trainer):
        """Test initialization."""
        assert trainer.model is not None
        assert trainer.use_bf16 is False
    
    def test_train_step(self, trainer):
        """Test training step."""
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        
        def loss_fn():
            x = torch.randn(10, 10)
            y = trainer.model(x)
            return y.mean()
        
        loss = trainer.train_step(loss_fn, optimizer)
        
        assert isinstance(loss, float)


class TestModelQuantizer:
    """Test Model Quantization."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    def test_get_model_size(self, model):
        """Test model size calculation."""
        size_info = ModelQuantizer.get_model_size(model)
        
        assert 'original_size_mb' in size_info
        assert 'quantized_size_mb' in size_info
        assert 'compression_ratio' in size_info
        assert size_info['compression_ratio'] >= 1.0
    
    def test_quantize_model(self, model):
        """Test model quantization."""
        calibration_data = [torch.randn(1, 10) for _ in range(50)]
        
        quantized = ModelQuantizer.quantize_model(model, calibration_data)
        
        assert quantized is not None


class TestModelPruner:
    """Test Model Pruning."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    def test_prune_model_magnitude(self, model):
        """Test magnitude pruning."""
        pruned = ModelPruner.prune_model(model, pruning_method="magnitude", sparsity=0.5)
        
        assert pruned is not None
        sparsity = ModelPruner.get_sparsity(pruned)
        assert sparsity >= 0.0
    
    def test_get_sparsity(self, model):
        """Test sparsity calculation."""
        sparsity = ModelPruner.get_sparsity(model)
        
        assert isinstance(sparsity, float)
        assert 0.0 <= sparsity <= 1.0


class TestKnowledgeDistillation:
    """Test Knowledge Distillation."""
    
    @pytest.fixture
    def teacher(self):
        return SimpleModel(input_dim=10, output_dim=2)
    
    @pytest.fixture
    def student(self):
        return SimpleModel(input_dim=10, output_dim=2)
    
    @pytest.fixture
    def distiller(self, teacher, student):
        return KnowledgeDistillation(teacher, student)
    
    def test_initialization(self, distiller):
        """Test initialization."""
        assert distiller.teacher_model is not None
        assert distiller.student_model is not None
        assert distiller.temperature > 0
    
    def test_distill(self, distiller):
        """Test distillation step."""
        optimizer = torch.optim.Adam(distiller.student_model.parameters(), lr=1e-3)
        data = torch.randn(10, 10)
        
        metrics = distiller.distill(data, optimizer)
        
        assert 'distillation_loss' in metrics
        assert 'total_loss' in metrics


class TestMetricsCollector:
    """Test Metrics Collector."""
    
    @pytest.fixture
    def collector(self, tmp_path):
        return MetricsCollector(tmp_path)
    
    def test_record_performance(self, collector):
        """Test recording performance metrics."""
        metrics = PerformanceMetrics(
            reward=10.0,
            wait_time=5.0,
            throughput=100.0,
            latency_ms=10.0,
            error_rate=0.01,
        )
        
        collector.record_performance(metrics)
        
        assert len(collector.performance_metrics) == 1
    
    def test_record_system(self, collector):
        """Test recording system metrics."""
        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_usage=70.0,
            inference_latency_ms=5.0,
            throughput_rps=200.0,
        )
        
        collector.record_system(metrics)
        
        assert len(collector.system_metrics) == 1
    
    def test_record_model(self, collector):
        """Test recording model metrics."""
        metrics = ModelMetrics(
            prediction_confidence=0.9,
            uncertainty=0.1,
            model_size_mb=10.0,
            sparsity=0.5,
        )
        
        collector.record_model(metrics)
        
        assert len(collector.model_metrics) == 1
    
    def test_save_metrics(self, collector, tmp_path):
        """Test saving metrics."""
        collector.record_performance(PerformanceMetrics(10.0, 5.0, 100.0, 10.0, 0.01))
        collector.save_metrics("test_metrics.json")
        
        assert (tmp_path / "test_metrics.json").exists()


class TestProductionOptimizer:
    """Test Production Optimizer."""
    
    @pytest.fixture
    def model(self):
        return SimpleModel()
    
    @pytest.fixture
    def optimizer(self, model):
        return ProductionOptimizer(
            model,
            use_mixed_precision=True,
            use_quantization=True,
            use_pruning=True,
        )
    
    def test_initialization(self, optimizer):
        """Test initialization."""
        assert optimizer.model is not None
        assert optimizer.use_mixed_precision == True
        assert optimizer.use_quantization == True
        assert optimizer.use_pruning == True
    
    def test_optimize_for_inference(self, optimizer):
        """Test inference optimization."""
        calibration_data = [torch.randn(1, 10) for _ in range(50)]
        
        optimized = optimizer.optimize_for_inference(calibration_data)
        
        assert optimized is not None
    
    def test_get_optimization_stats(self, optimizer):
        """Test getting optimization statistics."""
        stats = optimizer.get_optimization_stats()
        
        assert 'mixed_precision' in stats
        assert 'quantization' in stats
        assert 'pruning' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

