"""
Reproducibility Package Framework.

Tools for creating reproducible research packages including
code, data, and experiment configurations.
"""

import logging
import json
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available. YAML export will be disabled.")


@dataclass
class ExperimentSnapshot:
    """Snapshot of an experiment for reproducibility."""
    experiment_id: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    dataset: str
    results: Dict[str, float]
    model_path: Optional[str] = None
    code_version: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save snapshot to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Experiment snapshot saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExperimentSnapshot":
        """Load snapshot from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ReproducibilityPackage:
    """
    Reproducibility Package Creator.
    
    Creates comprehensive packages for reproducing research results,
    including code, data, configurations, and documentation.
    """
    
    def __init__(
        self,
        package_name: str,
        output_dir: str = "reproducibility_packages",
    ):
        """
        Initialize reproducibility package creator.
        
        Args:
            package_name: Name of the package
            output_dir: Output directory for packages
        """
        self.package_name = package_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.package_path: Optional[Path] = None
        self.experiments: List[ExperimentSnapshot] = []
        self.code_files: List[str] = []
        self.data_files: List[str] = []
    
    def create_package(self) -> Path:
        """Create package directory structure."""
        self.package_path = self.output_dir / self.package_name
        self.package_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.package_path / "code").mkdir(exist_ok=True)
        (self.package_path / "data").mkdir(exist_ok=True)
        (self.package_path / "experiments").mkdir(exist_ok=True)
        (self.package_path / "models").mkdir(exist_ok=True)
        (self.package_path / "results").mkdir(exist_ok=True)
        
        logger.info(f"Created package directory: {self.package_path}")
        return self.package_path
    
    def add_experiment(self, snapshot: ExperimentSnapshot) -> None:
        """Add experiment snapshot to package."""
        self.experiments.append(snapshot)
        
        # Save experiment snapshot
        if self.package_path:
            exp_path = self.package_path / "experiments" / f"{snapshot.experiment_id}.json"
            snapshot.save(str(exp_path))
    
    def add_code_file(self, source_path: str, target_path: Optional[str] = None) -> None:
        """Add code file to package."""
        if not self.package_path:
            self.create_package()
        
        source = Path(source_path)
        if not source.exists():
            logger.warning(f"Source file not found: {source_path}")
            return
        
        if target_path:
            target = self.package_path / "code" / target_path
        else:
            target = self.package_path / "code" / source.name
        
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        
        self.code_files.append(str(target.relative_to(self.package_path)))
        logger.info(f"Added code file: {target}")
    
    def add_data_file(self, source_path: str, target_path: Optional[str] = None) -> None:
        """Add data file to package."""
        if not self.package_path:
            self.create_package()
        
        source = Path(source_path)
        if not source.exists():
            logger.warning(f"Source file not found: {source_path}")
            return
        
        if target_path:
            target = self.package_path / "data" / target_path
        else:
            target = self.package_path / "data" / source.name
        
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        
        self.data_files.append(str(target.relative_to(self.package_path)))
        logger.info(f"Added data file: {target}")
    
    def create_readme(self, description: str = "") -> None:
        """Create README for package."""
        if not self.package_path:
            self.create_package()
        
        readme_content = f"""# {self.package_name}

## Description

{description or "Reproducibility package for research experiments."}

## Package Contents

- `code/`: Source code for experiments
- `data/`: Dataset files
- `experiments/`: Experiment configurations and snapshots
- `models/`: Trained model files
- `results/`: Experimental results

## Experiments

This package includes {len(self.experiments)} experiments:

"""
        
        for exp in self.experiments:
            readme_content += f"- **{exp.experiment_id}**: {exp.algorithm}\n"
            readme_content += f"  - Dataset: {exp.dataset}\n"
            readme_content += f"  - Results: {json.dumps(exp.results, indent=4)}\n\n"
        
        readme_content += f"""
## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run experiments:
   ```bash
   python scripts/run_experiments.py
   ```

3. View results in `results/` directory

## Reproducibility

All experiments were run with:
- Code version: See individual experiment snapshots
- Environment: See experiment configuration files
- Random seeds: Fixed for reproducibility

## Citation

If you use this package, please cite:
[Citation information]

## Contact

For questions or issues, please contact the research team.

---
Generated: {datetime.now().isoformat()}
"""
        
        readme_path = self.package_path / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        logger.info(f"Created README: {readme_path}")
    
    def create_requirements_file(self, requirements: List[str]) -> None:
        """Create requirements.txt file."""
        if not self.package_path:
            self.create_package()
        
        requirements_path = self.package_path / "requirements.txt"
        requirements_path.write_text('\n'.join(requirements) + '\n', encoding='utf-8')
        logger.info(f"Created requirements file: {requirements_path}")
    
    def generate_package_info(self) -> Dict[str, Any]:
        """Generate package information."""
        return {
            "package_name": self.package_name,
            "created_at": datetime.now().isoformat(),
            "num_experiments": len(self.experiments),
            "code_files": self.code_files,
            "data_files": self.data_files,
            "experiments": [exp.to_dict() for exp in self.experiments],
        }
    
    def save_package_info(self) -> None:
        """Save package information to JSON."""
        if not self.package_path:
            self.create_package()
        
        info = self.generate_package_info()
        info_path = self.package_path / "package_info.json"
        info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
        logger.info(f"Saved package info: {info_path}")


class CodeRelease:
    """
    Code Release Manager.
    
    Manages code releases for research publications,
    including versioning and documentation.
    """
    
    def __init__(
        self,
        release_name: str,
        version: str = "1.0.0",
    ):
        """
        Initialize code release.
        
        Args:
            release_name: Name of the release
            version: Version number
        """
        self.release_name = release_name
        self.version = version
        self.release_notes: List[str] = []
        self.features: List[str] = []
        self.bug_fixes: List[str] = []
    
    def add_release_note(self, note: str) -> None:
        """Add release note."""
        self.release_notes.append(note)
    
    def add_feature(self, feature: str) -> None:
        """Add feature description."""
        self.features.append(feature)
    
    def add_bug_fix(self, fix: str) -> None:
        """Add bug fix description."""
        self.bug_fixes.append(fix)
    
    def generate_changelog(self) -> str:
        """Generate changelog."""
        changelog = f"""# {self.release_name} - Version {self.version}

## Release Date
{datetime.now().strftime("%Y-%m-%d")}

## Features
"""
        for feature in self.features:
            changelog += f"- {feature}\n"
        
        if self.bug_fixes:
            changelog += "\n## Bug Fixes\n"
            for fix in self.bug_fixes:
                changelog += f"- {fix}\n"
        
        if self.release_notes:
            changelog += "\n## Notes\n"
            for note in self.release_notes:
                changelog += f"- {note}\n"
        
        return changelog
    
    def save_changelog(self, output_path: str) -> None:
        """Save changelog to file."""
        changelog = self.generate_changelog()
        Path(output_path).write_text(changelog, encoding='utf-8')
        logger.info(f"Changelog saved to {output_path}")

