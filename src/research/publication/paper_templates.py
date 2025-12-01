"""
Research Paper Templates.

Templates and utilities for creating research papers
and technical reports.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    title: str
    authors: List[str]
    affiliations: List[str]
    abstract: str
    keywords: List[str]
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    version: str = "1.0"
    doi: Optional[str] = None


class ResearchPaperTemplate:
    """
    Base template for research papers.
    
    Provides structure and utilities for creating research papers.
    """
    
    def __init__(
        self,
        metadata: PaperMetadata,
        template_format: str = "latex",  # or "markdown"
    ):
        """
        Initialize paper template.
        
        Args:
            metadata: Paper metadata
            template_format: Output format (latex, markdown)
        """
        self.metadata = metadata
        self.template_format = template_format
        self.sections: Dict[str, str] = {}
    
    def add_section(self, section_name: str, content: str) -> None:
        """Add a section to the paper."""
        self.sections[section_name] = content
    
    def generate_latex(self) -> str:
        """Generate LaTeX document."""
        latex = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}

\\title{{{self.metadata.title}}}
\\author{{{', '.join(self.metadata.authors)}}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{self.metadata.abstract}
\\end{{abstract}}

\\textbf{{Keywords:}} {', '.join(self.metadata.keywords)}

"""
        
        # Add sections
        for section_name, content in self.sections.items():
            latex += f"\\section{{{section_name}}}\n{content}\n\n"
        
        latex += "\\end{document}\n"
        
        return latex
    
    def generate_markdown(self) -> str:
        """Generate Markdown document."""
        md = f"""# {self.metadata.title}

**Authors:** {', '.join(self.metadata.authors)}  
**Date:** {self.metadata.date}  
**Version:** {self.metadata.version}

## Abstract

{self.metadata.abstract}

## Keywords

{', '.join(self.metadata.keywords)}

"""
        
        # Add sections
        for section_name, content in self.sections.items():
            md += f"## {section_name}\n\n{content}\n\n"
        
        return md
    
    def save(self, output_path: str) -> None:
        """Save paper to file."""
        output_file = Path(output_path)
        
        if self.template_format == "latex":
            content = self.generate_latex()
            output_file = output_file.with_suffix('.tex')
        else:
            content = self.generate_markdown()
            output_file = output_file.with_suffix('.md')
        
        output_file.write_text(content, encoding='utf-8')
        logger.info(f"Paper saved to {output_file}")


class ConferencePaperTemplate(ResearchPaperTemplate):
    """Template for conference papers."""
    
    def __init__(
        self,
        metadata: PaperMetadata,
        conference_name: str,
        template_format: str = "latex",
    ):
        """Initialize conference paper template."""
        super().__init__(metadata, template_format)
        self.conference_name = conference_name
    
    def generate_latex(self) -> str:
        """Generate conference paper LaTeX."""
        latex = f"""\\documentclass{{conference}}

\\title{{{self.metadata.title}}}
\\author{{{', '.join(self.metadata.authors)}}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{self.metadata.abstract}
\\end{{abstract}}

\\textbf{{Keywords:}} {', '.join(self.metadata.keywords)}

\\section{{Introduction}}

"""
        
        # Add sections
        for section_name, content in self.sections.items():
            latex += f"\\section{{{section_name}}}\n{content}\n\n"
        
        latex += "\\section{Conclusion}\n\n"
        latex += "\\bibliographystyle{plain}\n"
        latex += "\\bibliography{references}\n"
        latex += "\\end{document}\n"
        
        return latex


class JournalPaperTemplate(ResearchPaperTemplate):
    """Template for journal papers."""
    
    def __init__(
        self,
        metadata: PaperMetadata,
        journal_name: str,
        template_format: str = "latex",
    ):
        """Initialize journal paper template."""
        super().__init__(metadata, template_format)
        self.journal_name = journal_name
    
    def add_citation(self, citation: str) -> None:
        """Add citation to paper."""
        if "References" not in self.sections:
            self.sections["References"] = ""
        self.sections["References"] += f"{citation}\n\n"


def create_adaptive_traffic_paper_template(
    paper_type: str = "conference",
) -> ResearchPaperTemplate:
    """
    Create a pre-filled template for Adaptive Traffic Control paper.
    
    Args:
        paper_type: Type of paper ('conference' or 'journal')
        
    Returns:
        Pre-filled paper template
    """
    metadata = PaperMetadata(
        title="Adaptive Traffic Signal Control Using Deep Reinforcement Learning",
        authors=["Research Team"],
        affiliations=["Adaptive Traffic Control System"],
        abstract=(
            "This paper presents an advanced adaptive traffic signal control system "
            "using deep reinforcement learning with graph neural networks. "
            "The system achieves significant improvements in wait time reduction "
            "and traffic flow optimization through multi-agent coordination and "
            "real-time decision making."
        ),
        keywords=[
            "traffic control",
            "reinforcement learning",
            "graph neural networks",
            "adaptive systems",
            "intelligent transportation",
        ],
    )
    
    if paper_type == "conference":
        template = ConferencePaperTemplate(
            metadata=metadata,
            conference_name="IEEE Conference on Intelligent Transportation Systems",
        )
    else:
        template = JournalPaperTemplate(
            metadata=metadata,
            journal_name="IEEE Transactions on Intelligent Transportation Systems",
        )
    
    # Add standard sections
    template.add_section(
        "Introduction",
        "Traffic signal control is a critical component of urban transportation systems..."
    )
    template.add_section(
        "Related Work",
        "Previous work in adaptive traffic control has focused on..."
    )
    template.add_section(
        "Methodology",
        "Our approach combines deep reinforcement learning with graph neural networks..."
    )
    template.add_section(
        "Experiments",
        "We evaluate our system on synthetic and real-world traffic scenarios..."
    )
    template.add_section(
        "Results",
        "Our system achieves significant improvements over baseline methods..."
    )
    
    return template

