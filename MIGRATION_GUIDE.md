# Migration Guide: Updating Imports

This guide helps you update import statements from the old monolithic structure to the new modular structure.

## Import Mapping

### Core Components

**Old:**
```python
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.env.traffic_env import TrafficEnv
from src.env.sumo_env import SumoEnv
from src.env.marl_env import MarlEnv
from src.env.video_env import VideoTrafficEnv
from src.control.fuzzy_control import FuzzyController
from src.control.webster_method import WebsterMethod
from src.forecast.traffic_forecast import TrafficForecaster
from src.forecast.gnn_forecast import GNNForecaster
from src.optimization.genetic_algo import GeneticAlgorithm
from src.optimization.pso import ParticleSwarmOptimizer
```

**New:**
```python
from adaptive_traffic_core.rl.dqn_agent import DQNAgent, DQNConfig
from adaptive_traffic_core.env.traffic_env import TrafficEnv
from adaptive_traffic_core.env.sumo_env import SumoEnv
from adaptive_traffic_core.env.marl_env import MarlEnv
from adaptive_traffic_core.env.video_env import VideoTrafficEnv
from adaptive_traffic_core.control.fuzzy_control import FuzzyController
from adaptive_traffic_core.control.webster_method import WebsterMethod
from adaptive_traffic_core.forecast.traffic_forecast import TrafficForecaster
from adaptive_traffic_core.forecast.gnn_forecast import GNNForecaster
from adaptive_traffic_core.optimization.genetic_algo import GeneticAlgorithm
from adaptive_traffic_core.optimization.pso import ParticleSwarmOptimizer
```

### Vision Components

**Old:**
```python
from src.vision import VideoInputStream, VideoConfig, ROIManager, YOLOQueueEstimator, VideoSourceType
from src.vision.yolo_queue import process_frame_for_queues
```

**New:**
```python
from adaptive_traffic_vision.vision import VideoInputStream, VideoConfig, ROIManager, YOLOQueueEstimator, VideoSourceType
from adaptive_traffic_vision.vision.yolo_queue import process_frame_for_queues
```

### API Components

**Old:**
```python
from src.api.main import app
from src.api.routes.traffic import router
from src.api.services.traffic_controller import TrafficController
```

**New:**
```python
from adaptive_traffic_api.api.main import app
from adaptive_traffic_api.api.routes.traffic import router
from adaptive_traffic_api.api.services.traffic_controller import TrafficController
```

### Research Components

**Old:**
```python
from src.research.novel_algorithms.hierarchical_rl import HierarchicalRL
from src.research.federated_learning.federated_coordinator import FederatedCoordinator
from src.research.explainability.enhanced_explainability import EnhancedExplainability
```

**New:**
```python
from adaptive_traffic_research.research.novel_algorithms.hierarchical_rl import HierarchicalRL
from adaptive_traffic_research.research.federated_learning.federated_coordinator import FederatedCoordinator
from adaptive_traffic_research.research.explainability.enhanced_explainability import EnhancedExplainability
```

### Common/Utils Components

**Old:**
```python
from src.utils.config import load_config
from src.utils.metrics import calculate_metrics
from src.utils.health import health_check
from src.utils.io import save_image
from src.benchmarking.public_benchmark import PublicBenchmark
```

**New:**
```python
from adaptive_traffic_common.utils.config import load_config
from adaptive_traffic_common.utils.metrics import calculate_metrics
from adaptive_traffic_common.utils.health import health_check
from adaptive_traffic_common.utils.io import save_image
from adaptive_traffic_common.benchmarking.public_benchmark import PublicBenchmark
```

### Security Components

**Old:**
```python
from src.security.auth import authenticate_user
from src.security.rate_limiting import RateLimiter
```

**New:**
```python
from adaptive_traffic_api.security.auth import authenticate_user
from adaptive_traffic_api.security.rate_limiting import RateLimiter
```

## Automated Migration Script

You can use a find-and-replace approach to update imports. Here's a PowerShell script:

```powershell
# Update imports in Python files
$files = Get-ChildItem -Recurse -Filter "*.py"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    
    # Core imports
    $content = $content -replace 'from src\.rl\.', 'from adaptive_traffic_core.rl.'
    $content = $content -replace 'from src\.env\.', 'from adaptive_traffic_core.env.'
    $content = $content -replace 'from src\.control\.', 'from adaptive_traffic_core.control.'
    $content = $content -replace 'from src\.forecast\.', 'from adaptive_traffic_core.forecast.'
    $content = $content -replace 'from src\.optimization\.', 'from adaptive_traffic_core.optimization.'
    
    # Vision imports
    $content = $content -replace 'from src\.vision', 'from adaptive_traffic_vision.vision'
    
    # API imports
    $content = $content -replace 'from src\.api\.', 'from adaptive_traffic_api.api.'
    $content = $content -replace 'from src\.security\.', 'from adaptive_traffic_api.security.'
    
    # Research imports
    $content = $content -replace 'from src\.research\.', 'from adaptive_traffic_research.research.'
    
    # Common imports
    $content = $content -replace 'from src\.utils\.', 'from adaptive_traffic_common.utils.'
    $content = $content -replace 'from src\.benchmarking\.', 'from adaptive_traffic_common.benchmarking.'
    
    # Import src (direct)
    $content = $content -replace 'import src\.rl', 'import adaptive_traffic_core.rl'
    $content = $content -replace 'import src\.env', 'import adaptive_traffic_core.env'
    $content = $content -replace 'import src\.vision', 'import adaptive_traffic_vision.vision'
    $content = $content -replace 'import src\.api', 'import adaptive_traffic_api.api'
    
    Set-Content -Path $file.FullName -Value $content -NoNewline
}
```

## Step-by-Step Migration

1. **Install new packages:**
   ```bash
   pip install -e adaptive-traffic-common
   pip install -e adaptive-traffic-core
   pip install -e adaptive-traffic-api
   pip install -e adaptive-traffic-vision
   pip install -e adaptive-traffic-research
   ```

2. **Update imports in your code:**
   - Use the mapping above or run the migration script
   - Test each module after updating imports

3. **Update configuration paths:**
   - Config files are now in `adaptive-traffic-core/configs/`
   - Update any hardcoded paths

4. **Update test files:**
   - Test files should also use the new import paths
   - Update test fixtures and mocks

5. **Verify dependencies:**
   - Ensure all required packages are installed
   - Check that cross-project dependencies work

## Common Issues

### Issue: ModuleNotFoundError
**Solution:** Make sure you've installed the required sub-projects:
```bash
pip install -e adaptive-traffic-common
pip install -e adaptive-traffic-core
# etc.
```

### Issue: Circular Dependencies
**Solution:** Check that you're not creating circular imports between projects. Use `adaptive-traffic-common` for shared code.

### Issue: Config Paths
**Solution:** Update config paths to point to `adaptive-traffic-core/configs/` or use absolute paths.

## Testing After Migration

```bash
# Test core functionality
cd adaptive-traffic-core
python -m pytest tests/

# Test API
cd ../adaptive-traffic-api
python -m pytest tests/

# Test vision
cd ../adaptive-traffic-vision
python -m pytest tests/
```

## Need Help?

If you encounter issues during migration:
1. Check the individual project READMEs
2. Review the RESTRUCTURING_PLAN.md
3. Verify all packages are installed correctly
4. Check import paths match the new structure

