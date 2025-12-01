#!/usr/bin/env python3
"""Example script for adapting a new region using the intelligent recommendation system."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptation.adaptation_manager import AdaptationManager


def main():
    """Example usage of the adaptation system."""
    
    # Example checklist for a new region (e.g., a city in India)
    example_checklist = {
        "infrastructure": {
            "power": {
                "availability_hours": 22.0,  # Occasional outages
                "outage_frequency": "occasional",
                "backup_available": True
            },
            "network": {
                "type": "4g",
                "bandwidth_mbps": 25.0,
                "reliability_percent": 96.0,
                "latency_ms": 50.0
            },
            "hardware": {
                "cpu_cores": 4,
                "ram_gb": 8,
                "gpu_available": False,
                "storage_gb": 128
            },
            "cameras": {
                "quality": "1080p",
                "existing_cctv": True,
                "ip_cameras": True
            }
        },
        "traffic": {
            "patterns": {
                "peak_hours_defined": True,
                "historical_data_months": 2,
                "predictable": False,
                "seasonal_variations": True,
                "special_events_frequent": True
            },
            "volume": {
                "average_queue_length": 35.0,
                "max_queue_length": 60.0,
                "peak_hour_vehicles_per_hour": 1200
            },
            "vehicle_mix_diverse": True,
            "lane_discipline_good": False
        },
        "deployment": {
            "num_intersections": 5,
            "budget_per_intersection": 20000.0,
            "explainability_required": True,
            "regulatory_compliance_strict": False,
            "maintenance_team_available": True,
            "ml_team_available": False,
            "edge_deployment_preferred": True,
            "multi_intersection_coordination": False,
            "real_time_critical": True
        },
        "custom_requirements": {}
    }
    
    # Initialize adaptation manager
    manager = AdaptationManager()
    
    # Run adaptation
    print("="*80)
    print("INTELLIGENT REGIONAL ADAPTATION SYSTEM")
    print("="*80)
    print("\nAdapting region based on checklist requirements...\n")
    
    report = manager.adapt_region(
        checklist_data=example_checklist,
        region_name="Example City",
        output_dir="./adaptation_outputs"
    )
    
    # Display summary
    print("\n" + "="*80)
    print("ADAPTATION SUMMARY")
    print("="*80)
    print(f"\nRegion: {report['region_name']}")
    print(f"\nRecommended Technology Stack:")
    print(f"  Control:      {report['summary']['recommended_control'].upper()}")
    print(f"  Vision:       {report['summary']['recommended_vision'].upper()}")
    print(f"  Forecasting:  {report['summary']['recommended_forecasting'].upper()}")
    print(f"  Deployment:   {report['summary']['recommended_deployment'].upper()}")
    
    print(f"\nMetrics:")
    print(f"  Confidence:   {report['summary']['confidence_score']:.1%}")
    print(f"  Cost:         ${report['summary']['estimated_cost']:,.0f} per intersection")
    print(f"  Wait Time:    {report['summary']['estimated_wait_time']:.1f} seconds")
    
    print(f"\nReasoning:")
    for reason in report['recommendation']['reasoning']:
        print(f"  • {reason}")
    
    print(f"\nNext Steps:")
    for step in report['next_steps']:
        print(f"  {step}")
    
    print(f"\n" + "="*80)
    print(f"Full report and configuration saved to: ./adaptation_outputs/")
    print("="*80)


if __name__ == '__main__':
    main()

