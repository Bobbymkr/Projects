#!/usr/bin/env python3
"""
Track budget spending for the Perfect Score Execution Plan.

Usage:
    python scripts/track_budget.py --week 1 --spent 15000 --budget 17000
    python scripts/track_budget.py --report --output budget_report.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
BUDGET_DIR = PROJECT_ROOT / "execution" / "budget"

# Budget allocation from execution plan
BUDGET_ALLOCATION = {
    0: {"phase": "Preparation", "budget": 2000, "cumulative": 2000},
    1: {"phase": "Performance Validation", "budget": 15000, "cumulative": 17000},
    2: {"phase": "Performance Validation", "budget": 0, "cumulative": 17000},  # Week 1-2 combined
    3: {"phase": "Testing Infrastructure", "budget": 18000, "cumulative": 35000},
    4: {"phase": "Testing Infrastructure", "budget": 0, "cumulative": 35000},  # Week 3-4 combined
    5: {"phase": "Production Monitoring", "budget": 22000, "cumulative": 57000},
    6: {"phase": "Production Monitoring", "budget": 0, "cumulative": 57000},  # Week 5-6 combined
    7: {"phase": "Auto-Scaling & HA", "budget": 12000, "cumulative": 69000},
    8: {"phase": "Auto-Scaling & HA", "budget": 0, "cumulative": 69000},  # Week 7-8 combined
    9: {"phase": "Architecture Enhancement", "budget": 8000, "cumulative": 77000},
    10: {"phase": "Architecture Enhancement", "budget": 0, "cumulative": 77000},  # Week 9-10 combined
    11: {"phase": "Validation & Polish", "budget": 3000, "cumulative": 80000},
    12: {"phase": "Validation & Polish", "budget": 0, "cumulative": 80000},  # Week 11-13 combined
    13: {"phase": "Validation & Polish", "budget": 0, "cumulative": 80000},
}

TOTAL_BUDGET = 80000


def get_week_budget(week: int) -> int:
    """Get budget allocation for a week."""
    if week in BUDGET_ALLOCATION:
        return BUDGET_ALLOCATION[week]["budget"]
    return 0


def get_cumulative_budget(week: int) -> int:
    """Get cumulative budget up to a week."""
    if week in BUDGET_ALLOCATION:
        return BUDGET_ALLOCATION[week]["cumulative"]
    # Calculate if not in allocation
    cumulative = 0
    for w in range(week + 1):
        if w in BUDGET_ALLOCATION:
            cumulative = BUDGET_ALLOCATION[w]["cumulative"]
    return cumulative


def calculate_burn_rate(spent: float, budget: float) -> float:
    """Calculate burn rate percentage."""
    if budget == 0:
        return 0.0
    return (spent / budget) * 100


def get_burn_rate_status(burn_rate: float) -> tuple:
    """Get burn rate status and emoji."""
    if burn_rate > 120:
        return "🚨 Critical", "critical"
    elif burn_rate > 110:
        return "⚠️ Warning", "warning"
    elif burn_rate >= 90:
        return "✅ On Track", "on_track"
    else:
        return "📉 Under Budget", "under_budget"


def track_budget(week: int, spent: float, budget: Optional[float] = None) -> Dict:
    """Track budget for a specific week."""
    if budget is None:
        budget = get_week_budget(week)
    
    cumulative_budget = get_cumulative_budget(week)
    burn_rate = calculate_burn_rate(spent, budget) if budget > 0 else 0.0
    status, status_type = get_burn_rate_status(burn_rate)
    
    budget_data = {
        "week": week,
        "timestamp": datetime.now().isoformat(),
        "phase": BUDGET_ALLOCATION.get(week, {}).get("phase", "Unknown"),
        "budget": {
            "allocated": budget,
            "spent": spent,
            "remaining": budget - spent,
            "burn_rate": burn_rate,
            "status": status,
            "status_type": status_type
        },
        "cumulative": {
            "budget": cumulative_budget,
            "spent": spent,  # Would need to sum all previous weeks
            "percentage": (cumulative_budget / TOTAL_BUDGET * 100) if TOTAL_BUDGET > 0 else 0
        }
    }
    
    # Save to file
    BUDGET_DIR.mkdir(parents=True, exist_ok=True)
    budget_file = BUDGET_DIR / f"budget_week{week}.json"
    with open(budget_file, 'w') as f:
        json.dump(budget_data, f, indent=2)
    
    return budget_data


def generate_report(output: Optional[Path] = None) -> Dict:
    """Generate budget report."""
    # Load all budget files
    budget_files = sorted(BUDGET_DIR.glob("budget_week*.json"))
    
    weekly_data = []
    total_spent = 0.0
    
    for budget_file in budget_files:
        with open(budget_file) as f:
            data = json.load(f)
            weekly_data.append(data)
            total_spent += data["budget"]["spent"]
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_budget": TOTAL_BUDGET,
        "total_spent": total_spent,
        "remaining": TOTAL_BUDGET - total_spent,
        "overall_burn_rate": calculate_burn_rate(total_spent, TOTAL_BUDGET),
        "weekly_data": weekly_data,
        "summary": {
            "weeks_tracked": len(weekly_data),
            "average_weekly_spend": total_spent / len(weekly_data) if weekly_data else 0,
            "projected_total": (total_spent / len(weekly_data) * 13) if weekly_data else 0,
        }
    }
    
    # Add overall status
    overall_status, _ = get_burn_rate_status(report["overall_burn_rate"])
    report["overall_status"] = overall_status
    
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Budget report saved to {output}")
    else:
        # Print to console
        print("\n💰 Budget Report")
        print("=" * 80)
        print(f"Total Budget: ${TOTAL_BUDGET:,.2f}")
        print(f"Total Spent: ${total_spent:,.2f}")
        print(f"Remaining: ${report['remaining']:,.2f}")
        print(f"Burn Rate: {report['overall_burn_rate']:.2f}%")
        print(f"Status: {overall_status}")
        print("\nWeekly Breakdown:")
        print("-" * 80)
        for data in weekly_data:
            week = data["week"]
            spent = data["budget"]["spent"]
            budget = data["budget"]["allocated"]
            burn_rate = data["budget"]["burn_rate"]
            status = data["budget"]["status"]
            print(f"Week {week:2d}: ${spent:>10,.2f} / ${budget:>10,.2f} ({burn_rate:>6.2f}%) {status}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Track budget spending")
    parser.add_argument("--week", type=int, help="Week number (0-13)")
    parser.add_argument("--spent", type=float, help="Amount spent this week")
    parser.add_argument("--budget", type=float, help="Budget for this week (optional)")
    parser.add_argument("--report", action="store_true", help="Generate budget report")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.report:
        generate_report(output=args.output)
    elif args.week is not None and args.spent is not None:
        budget_data = track_budget(args.week, args.spent, args.budget)
        
        # Print summary
        print(f"\n💰 Budget Tracking - Week {args.week}")
        print("=" * 60)
        print(f"Phase: {budget_data['phase']}")
        print(f"Spent: ${budget_data['budget']['spent']:,.2f}")
        print(f"Budget: ${budget_data['budget']['allocated']:,.2f}")
        print(f"Remaining: ${budget_data['budget']['remaining']:,.2f}")
        print(f"Burn Rate: {budget_data['budget']['burn_rate']:.2f}%")
        print(f"Status: {budget_data['budget']['status']}")
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

