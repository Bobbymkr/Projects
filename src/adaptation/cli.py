"""Command-line interface for regional adaptation system."""

import argparse
import json
import sys
from pathlib import Path

from .adaptation_manager import AdaptationManager
from .checklist_parser import ChecklistParser


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Regional Adaptation System for Adaptive Traffic Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate recommendations from checklist file
  python -m src.adaptation.cli recommend checklist.json --region "Mumbai, India"
  
  # Create template checklist
  python -m src.adaptation.cli template --output my_checklist.json
  
  # Generate full adaptation (recommendations + config)
  python -m src.adaptation.cli adapt checklist.json --output-dir ./outputs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Generate technology recommendations')
    recommend_parser.add_argument('checklist', help='Path to checklist JSON file')
    recommend_parser.add_argument('--region', default='new_region', help='Region name')
    recommend_parser.add_argument('--output', help='Output file path (optional)')
    
    # Adapt command
    adapt_parser = subparsers.add_parser('adapt', help='Complete adaptation workflow')
    adapt_parser.add_argument('checklist', help='Path to checklist JSON file')
    adapt_parser.add_argument('--region', help='Region name (defaults to filename)')
    adapt_parser.add_argument('--output-dir', default='./adaptation_outputs', 
                             help='Output directory for reports and configs')
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Generate checklist template')
    template_parser.add_argument('--output', default='checklist_template.json',
                                help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'recommend':
            handle_recommend(args)
        elif args.command == 'adapt':
            handle_adapt(args)
        elif args.command == 'template':
            handle_template(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_recommend(args):
    """Handle recommend command."""
    manager = AdaptationManager()
    
    # Parse checklist
    checklist_path = Path(args.checklist)
    if not checklist_path.exists():
        print(f"Error: Checklist file not found: {args.checklist}", file=sys.stderr)
        sys.exit(1)
    
    requirements = manager.parser.parse_from_file(str(checklist_path))
    
    # Generate recommendations
    tech_stack = manager.recommender.recommend(requirements)
    
    # Display results
    print("\n" + "="*80)
    print("TECHNOLOGY RECOMMENDATION REPORT")
    print("="*80)
    print(f"\nRegion: {args.region}")
    print(f"\nRecommended Technology Stack:")
    print(f"  Control Algorithm:    {tech_stack.control_algorithm.value.upper()}")
    print(f"  Vision Model:         {tech_stack.vision_model.value.upper()}")
    print(f"  Forecasting Model:   {tech_stack.forecasting_model.value.upper()}")
    print(f"  Deployment:           {tech_stack.deployment_architecture.value.upper()}")
    print(f"\nConfidence Score: {tech_stack.confidence_score:.1%}")
    print(f"Estimated Cost:   ${tech_stack.estimated_cost:,.0f} per intersection")
    print(f"Estimated Wait:   {tech_stack.estimated_performance.get('estimated_wait_time_seconds', 0):.1f} seconds")
    
    print(f"\nReasoning:")
    for reason in tech_stack.reasoning:
        print(f"  • {reason}")
    
    if tech_stack.alternatives:
        print(f"\nAlternative Options:")
        for i, alt in enumerate(tech_stack.alternatives, 1):
            print(f"\n  Alternative {i}:")
            print(f"    Control: {alt.control_algorithm.value}")
            print(f"    Vision: {alt.vision_model.value}")
            print(f"    Cost: ${alt.estimated_cost:,.0f}")
            print(f"    Confidence: {alt.confidence_score:.1%}")
    
    # Save if output specified
    if args.output:
        output = {
            "region": args.region,
            "recommendation": tech_stack.to_dict()
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nRecommendation saved to {args.output}")


def handle_adapt(args):
    """Handle adapt command."""
    manager = AdaptationManager()
    
    region_name = args.region or Path(args.checklist).stem
    
    print(f"\nStarting adaptation workflow for {region_name}...")
    print("="*80)
    
    report = manager.adapt_from_file(
        args.checklist,
        region_name=region_name,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("ADAPTATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - {region_name}_adaptation_report.json")
    print(f"  - {region_name}_config.json")
    print(f"  - {region_name}_summary.txt")
    
    print(f"\nSummary:")
    print(f"  Recommended Stack: {report['summary']['recommended_control']} + "
          f"{report['summary']['recommended_vision']}")
    print(f"  Confidence: {report['summary']['confidence_score']:.1%}")
    print(f"  Cost: ${report['summary']['estimated_cost']:,.0f}")
    print(f"  Wait Time: {report['summary']['estimated_wait_time']:.1f}s")
    
    print(f"\nNext Steps:")
    for step in report['next_steps'][:3]:
        print(f"  {step}")


def handle_template(args):
    """Handle template command."""
    parser = ChecklistParser()
    template = parser.create_template()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"Checklist template created: {output_path}")
    print("\nFill in the checklist with your regional requirements and use:")
    print(f"  python -m src.adaptation.cli adapt {output_path}")


if __name__ == '__main__':
    main()

