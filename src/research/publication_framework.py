"""
Research Publication Framework.

Tools for creating reproducible research publications with code,
data, and documentation.
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class PublicationPackage:
    """
    Research Publication Package Generator.
    
    Creates reproducible packages for research publications.
    """
    
    def __init__(
        self,
        paper_title: str,
        authors: List[str],
        output_dir: Path,
    ):
        """Initialize publication package."""
        self.paper_title = paper_title
        self.authors = authors
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_package(
        self,
        code_files: List[Path],
        data_files: Optional[List[Path]] = None,
        model_files: Optional[List[Path]] = None,
        config_files: Optional[List[Path]] = None,
    ) -> Path:
        """
        Create complete publication package.
        
        Args:
            code_files: List of code files to include
            data_files: List of data files to include
            model_files: List of model files to include
            config_files: List of config files to include
            
        Returns:
            Path to created package
        """
        logger.info(f"Creating publication package: {self.paper_title}")
        
        # Create directory structure
        code_dir = self.output_dir / "code"
        data_dir = self.output_dir / "data"
        models_dir = self.output_dir / "models"
        config_dir = self.output_dir / "configs"
        docs_dir = self.output_dir / "docs"
        
        for dir_path in [code_dir, data_dir, models_dir, config_dir, docs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Copy files
        self._copy_files(code_files, code_dir)
        if data_files:
            self._copy_files(data_files, data_dir)
        if model_files:
            self._copy_files(model_files, models_dir)
        if config_files:
            self._copy_files(config_files, config_dir)
        
        # Create metadata
        self._create_metadata()
        
        # Create README
        self._create_readme()
        
        # Create requirements file
        self._create_requirements()
        
        logger.info(f"Package created at: {self.output_dir}")
        return self.output_dir
    
    def _copy_files(self, files: List[Path], target_dir: Path):
        """Copy files to target directory."""
        for file_path in files:
            if file_path.exists():
                shutil.copy2(file_path, target_dir / file_path.name)
    
    def _create_metadata(self):
        """Create metadata file."""
        metadata = {
            "paper_title": self.paper_title,
            "authors": self.authors,
            "created_date": datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "Reproducible research package for traffic control system",
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_readme(self):
        """Create README for package."""
        readme_content = f"""# {self.paper_title}

## Authors
{', '.join(self.authors)}

## Description
This package contains code, data, and models for reproducing the results
presented in the research paper.

## Structure
- `code/`: Source code for algorithms and experiments
- `data/`: Datasets used in experiments
- `models/`: Trained models
- `configs/`: Configuration files
- `docs/`: Additional documentation

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run experiments: See `docs/experiments.md`

## Reproducibility
All random seeds are fixed for reproducibility.
See `configs/experiment_config.json` for details.

## Citation
If you use this code, please cite:
```
[Citation information]
```

## License
[License information]
"""
        
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def _create_requirements(self):
        """Create requirements file."""
        requirements = """numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
"""
        
        req_file = self.output_dir / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write(requirements)


class ExperimentReproducer:
    """
    Experiment Reproducer.
    
    Ensures experiments can be reproduced with fixed seeds and configurations.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize reproducer with fixed seed."""
        self.seed = seed
        self.set_seeds()
    
    def set_seeds(self):
        """Set all random seeds for reproducibility."""
        np.random.seed(self.seed)
        # In production, also set PyTorch, TensorFlow seeds
    
    def save_experiment_config(
        self,
        config: Dict[str, Any],
        output_path: Path,
    ):
        """Save experiment configuration."""
        config["seed"] = self.seed
        config["timestamp"] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_experiment_config(self, config_path: Path) -> Dict[str, Any]:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Restore seed
        if "seed" in config:
            self.seed = config["seed"]
            self.set_seeds()
        
        return config


class PaperTemplate:
    """Research paper template generator."""
    
    @staticmethod
    def generate_latex_template(
        title: str,
        authors: List[str],
        abstract: str,
        sections: List[str],
    ) -> str:
        """Generate LaTeX paper template."""
        authors_str = " \\and ".join([f"\\author{{{author}}}" for author in authors])
        
        template = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}

\\title{{{title}}}
{authors_str}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
[Introduction content]

\\section{{Methodology}}
[Methodology content]

\\section{{Experiments}}
[Experiments content]

\\section{{Results}}
[Results content]

\\section{{Conclusion}}
[Conclusion content]

\\bibliography{{references}}
\\end{{document}}
"""
        return template
    
    @staticmethod
    def generate_markdown_template(
        title: str,
        authors: List[str],
        abstract: str,
    ) -> str:
        """Generate Markdown paper template."""
        template = f"""# {title}

**Authors**: {', '.join(authors)}

**Date**: {datetime.now().strftime('%Y-%m-%d')}

## Abstract

{abstract}

## 1. Introduction

[Introduction content]

## 2. Methodology

[Methodology content]

## 3. Experiments

[Experiments content]

## 4. Results

[Results content]

## 5. Conclusion

[Conclusion content]

## References

[References]
"""
        return template

