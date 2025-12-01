"""Adaptation Manager - Main Interface for Regional Adaptation.

Provides a unified interface for the complete adaptation workflow:
1. Parse checklist
2. Generate recommendations
3. Generate configuration
4. Export results
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .checklist_parser import ChecklistParser, RegionalRequirements
from .recommendation_engine import TechnologyRecommendationEngine, TechnologyStack
from .config_generator import RegionalConfigGenerator

logger = logging.getLogger(__name__)


class AdaptationManager:
    """Main manager for regional adaptation workflow."""
    
    def __init__(self):
        """Initialize the adaptation manager."""
        self.logger = logging.getLogger(__name__)
        self.parser = ChecklistParser()
        self.recommender = TechnologyRecommendationEngine()
        self.generator = RegionalConfigGenerator()
    
    def adapt_region(self, checklist_data: Dict[str, Any], 
                    region_name: str = "new_region",
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Complete adaptation workflow for a new region.
        
        Args:
            checklist_data: Checklist selections as dictionary
            region_name: Name of the region
            output_dir: Directory to save outputs (optional)
            
        Returns:
            Complete adaptation report with recommendations and configuration
        """
        self.logger.info(f"Starting adaptation workflow for {region_name}...")
        
        # Step 1: Parse checklist
        self.logger.info("Step 1: Parsing checklist...")
        requirements = self.parser.parse_from_dict(checklist_data)
        
        # Step 2: Generate recommendations
        self.logger.info("Step 2: Generating technology recommendations...")
        tech_stack = self.recommender.recommend(requirements)
        
        # Step 3: Generate configuration
        self.logger.info("Step 3: Generating regional configuration...")
        config = self.generator.generate(requirements, tech_stack, region_name)
        
        # Step 4: Create report
        report = self._create_report(requirements, tech_stack, config, region_name)
        
        # Step 5: Save outputs if directory provided
        if output_dir:
            self._save_outputs(output_dir, region_name, report, config, tech_stack)
        
        self.logger.info(f"Adaptation workflow completed for {region_name}")
        
        return report
    
    def adapt_from_file(self, checklist_path: str,
                       region_name: Optional[str] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Adapt region from checklist file.
        
        Args:
            checklist_path: Path to checklist JSON file
            region_name: Name of region (defaults to filename)
            output_dir: Directory to save outputs
            
        Returns:
            Complete adaptation report
        """
        if region_name is None:
            region_name = Path(checklist_path).stem
        
        checklist_data = self.parser.parse_from_file(checklist_path)
        requirements = self.parser.parse_from_dict(checklist_data)
        
        # Convert requirements back to dict for adapt_region
        # (We need the original checklist data)
        with open(checklist_path, 'r') as f:
            checklist_dict = json.load(f)
        
        return self.adapt_region(checklist_dict, region_name, output_dir)
    
    def _create_report(self, requirements: RegionalRequirements,
                      tech_stack: TechnologyStack,
                      config: Dict[str, Any],
                      region_name: str) -> Dict[str, Any]:
        """Create comprehensive adaptation report."""
        report = {
            "region_name": region_name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "recommended_control": tech_stack.control_algorithm.value,
                "recommended_vision": tech_stack.vision_model.value,
                "recommended_forecasting": tech_stack.forecasting_model.value,
                "recommended_deployment": tech_stack.deployment_architecture.value,
                "confidence_score": tech_stack.confidence_score,
                "estimated_cost": tech_stack.estimated_cost,
                "estimated_wait_time": tech_stack.estimated_performance.get("estimated_wait_time_seconds", 0)
            },
            "requirements_analysis": {
                "infrastructure_level": requirements.infrastructure.get_infrastructure_level().value,
                "traffic_density": requirements.traffic.get_traffic_density().value,
                "traffic_pattern": requirements.traffic.get_traffic_pattern().value,
                "num_intersections": requirements.num_intersections,
                "budget_per_intersection": requirements.budget_per_intersection
            },
            "recommendation": {
                "primary_stack": tech_stack.to_dict(),
                "reasoning": tech_stack.reasoning,
                "alternatives": [alt.to_dict() for alt in tech_stack.alternatives]
            },
            "configuration": config,
            "next_steps": self._generate_next_steps(requirements, tech_stack)
        }
        
        return report
    
    def _generate_next_steps(self, requirements: RegionalRequirements,
                           tech_stack: TechnologyStack) -> list:
        """Generate actionable next steps."""
        steps = []
        
        steps.append("1. Review the recommended technology stack and reasoning")
        
        if tech_stack.control_algorithm.value == "dqn" and not requirements.ml_team_available:
            steps.append("2. ⚠️  WARNING: DQN requires ML expertise - consider Fuzzy Logic alternative")
        
        if tech_stack.estimated_cost > requirements.budget_per_intersection:
            steps.append(f"3. ⚠️  Budget exceeded - consider alternatives or increase budget")
        
        steps.append("4. Validate configuration parameters with local traffic engineers")
        steps.append("5. Set up pilot deployment at 1-2 test intersections")
        steps.append("6. Monitor performance for 2-4 weeks and fine-tune parameters")
        
        if tech_stack.forecasting_model.value != "none":
            steps.append("7. Collect and prepare historical traffic data for forecasting")
        
        if tech_stack.control_algorithm.value in ["dqn", "marl"]:
            steps.append("8. Plan training phase with sufficient data collection period")
        
        steps.append("9. Scale to additional intersections based on pilot results")
        
        return steps
    
    def _save_outputs(self, output_dir: str, region_name: str,
                     report: Dict[str, Any], config: Dict[str, Any],
                     tech_stack: TechnologyStack) -> None:
        """Save all outputs to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = output_path / f"{region_name}_adaptation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Report saved to {report_path}")
        
        # Save configuration
        config_path = output_path / f"{region_name}_config.json"
        self.generator.save_config(config, str(config_path))
        
        # Save summary
        summary_path = output_path / f"{region_name}_summary.txt"
        self._save_summary(summary_path, report, tech_stack)
    
    def _save_summary(self, summary_path: Path, report: Dict[str, Any],
                     tech_stack: TechnologyStack) -> None:
        """Save human-readable summary."""
        summary = f"""
================================================================================
REGIONAL ADAPTATION SUMMARY
================================================================================

Region: {report['region_name']}
Generated: {report['generated_at']}

RECOMMENDED TECHNOLOGY STACK
--------------------------------------------------------------------------------
Control Algorithm:    {tech_stack.control_algorithm.value.upper()}
Vision Model:         {tech_stack.vision_model.value.upper()}
Forecasting Model:    {tech_stack.forecasting_model.value.upper()}
Deployment:           {tech_stack.deployment_architecture.value.upper()}

CONFIDENCE & COST
--------------------------------------------------------------------------------
Confidence Score:     {tech_stack.confidence_score:.1%}
Estimated Cost:       ${tech_stack.estimated_cost:,.0f} per intersection
Estimated Wait Time:  {tech_stack.estimated_performance.get('estimated_wait_time_seconds', 0):.1f} seconds

REQUIREMENTS ANALYSIS
--------------------------------------------------------------------------------
Infrastructure Level: {report['requirements_analysis']['infrastructure_level']}
Traffic Density:      {report['requirements_analysis']['traffic_density']}
Traffic Pattern:      {report['requirements_analysis']['traffic_pattern']}
Intersections:        {report['requirements_analysis']['num_intersections']}
Budget:               ${report['requirements_analysis']['budget_per_intersection']:,.0f}

REASONING
--------------------------------------------------------------------------------
"""
        for reason in tech_stack.reasoning:
            summary += f"• {reason}\n"
        
        summary += f"""
NEXT STEPS
--------------------------------------------------------------------------------
"""
        for step in report['next_steps']:
            summary += f"{step}\n"
        
        if tech_stack.alternatives:
            summary += f"""
ALTERNATIVE OPTIONS
--------------------------------------------------------------------------------
"""
            for i, alt in enumerate(tech_stack.alternatives, 1):
                summary += f"\nAlternative {i}:\n"
                summary += f"  Control: {alt.control_algorithm.value}\n"
                summary += f"  Vision: {alt.vision_model.value}\n"
                summary += f"  Cost: ${alt.estimated_cost:,.0f}\n"
                summary += f"  Confidence: {alt.confidence_score:.1%}\n"
        
        summary += "\n================================================================================\n"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.logger.info(f"Summary saved to {summary_path}")

