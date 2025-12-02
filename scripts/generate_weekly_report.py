#!/usr/bin/env python3
"""
Generate weekly progress report.

Usage:
    python scripts/generate_weekly_report.py --week 1 --output weekly_report_week1.md
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
PROGRESS_DIR = PROJECT_ROOT / "execution" / "progress"
REPORTS_DIR = PROJECT_ROOT / "execution" / "reports"


def load_progress(week: int) -> Optional[Dict]:
    """Load progress data for a week."""
    progress_file = PROGRESS_DIR / f"progress_week{week}.json"
    if not progress_file.exists():
        return None
    
    with open(progress_file) as f:
        return json.load(f)


def load_budget(week: int) -> Optional[Dict]:
    """Load budget data for a week."""
    budget_file = PROJECT_ROOT / "execution" / "budget" / f"budget_week{week}.json"
    if not budget_file.exists():
        return None
    
    with open(budget_file) as f:
        return json.load(f)


def generate_markdown_report(week: int, output: Path) -> str:
    """Generate markdown weekly report."""
    progress = load_progress(week)
    budget = load_budget(week)
    
    report = f"""# Weekly Progress Report - Week {week}

**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Week**: {week}  
**Status**: In Progress

---

## 📊 Executive Summary

"""
    
    if progress:
        overall = progress.get("overall_score", {})
        report += f"""
- **Overall Progress**: {overall.get('progress', 0):.2f}%
- **Current Score**: {overall.get('current', 92):.1f}/100
- **Target Score**: {overall.get('target', 100):.1f}/100

"""
    
    report += """
## 📈 Key Metrics

"""
    
    if progress:
        metrics = progress.get("metrics", {})
        
        for metric_name, metric_data in metrics.items():
            status = metric_data.get("status", "❓")
            metric_title = metric_name.replace("_", " ").title()
            
            report += f"### {status} {metric_title}\n\n"
            
            if "current" in metric_data:
                report += f"- **Current**: {metric_data['current']:.2f}%\n"
                report += f"- **Target**: {metric_data.get('target', 'N/A')}\n"
            elif "percentage" in metric_data:
                report += f"- **Completed**: {metric_data['completed']}/{metric_data['total']}\n"
                report += f"- **Percentage**: {metric_data['percentage']:.2f}%\n"
            elif "score" in metric_data:
                report += f"- **Score**: {metric_data['score']:.2f}%\n"
            
            report += "\n"
    
    if budget:
        report += """
## 💰 Budget Status

"""
        budget_data = budget.get("budget", {})
        report += f"""
- **Spent**: ${budget_data.get('spent', 0):,.2f}
- **Budget**: ${budget_data.get('allocated', 0):,.2f}
- **Remaining**: ${budget_data.get('remaining', 0):,.2f}
- **Burn Rate**: {budget_data.get('burn_rate', 0):.2f}%
- **Status**: {budget_data.get('status', 'Unknown')}

"""
    
    report += """
## ✅ Completed This Week

- [ ] Add completed tasks here

## 🚧 In Progress

- [ ] Add in-progress tasks here

## ⚠️ Blockers & Risks

- [ ] Add blockers and risks here

## 📋 Next Week Plan

- [ ] Add next week's plan here

---

*Report generated automatically by generate_weekly_report.py*
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate weekly progress report")
    parser.add_argument("--week", type=int, required=True, help="Week number (0-13)")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.output:
        output_path = args.output
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"weekly_report_week{args.week}.md"
    
    report = generate_markdown_report(args.week, output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Weekly report generated: {output_path}")


if __name__ == "__main__":
    main()

