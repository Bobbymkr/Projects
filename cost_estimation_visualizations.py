#!/usr/bin/env python3
"""
Cost Estimation Visualization Tool
Creates charts and graphs for the comprehensive cost analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_analysis_data():
    """Load the comprehensive cost analysis data"""
    try:
        with open('comprehensive_cost_analysis.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Run costar_system_star_analysis.py first to generate data")
        return None

def create_function_point_breakdown(data):
    """Create Function Point breakdown chart"""
    fp_data = data['function_points']['ufp_breakdown']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of FP breakdown
    categories = list(fp_data.keys())
    values = list(fp_data.values())
    
    bars = ax1.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_title('Function Point Breakdown', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Function Points')
    ax1.set_xlabel('Function Categories')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart of FP distribution
    ax2.pie(values, labels=categories, autopct='%1.1f%%', startangle=90,
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax2.set_title('Function Point Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('function_point_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_cost_comparison_chart(data):
    """Create cost comparison chart across different methods"""
    
    methods = ['COCOMO Base', 'With Adjustments', 'Monte Carlo Mean', 'Monte Carlo 90th %ile']
    costs = [
        data['costs']['total_project_cost'],
        data['costs']['total_project_cost'],
        data['advanced']['monte_carlo']['mean_cost'],
        data['advanced']['monte_carlo']['percentile_90']
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(methods, costs, color=['#3498db', '#e74c3c', '#f39c12', '#e67e22'])
    
    ax.set_title('Cost Estimation Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cost (USD)', fontsize=12)
    ax.set_xlabel('Estimation Method', fontsize=12)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
                f'${cost/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_risk_analysis_chart(data):
    """Create risk analysis visualization"""
    
    risk_data = data['costar']['risk_analysis']['detailed_risks']
    risks = list(risk_data.keys())
    impacts = list(risk_data.values())
    
    # Create risk levels
    risk_levels = []
    colors = []
    for impact in impacts:
        if impact > 1.3:
            risk_levels.append('HIGH')
            colors.append('#e74c3c')
        elif impact > 1.1:
            risk_levels.append('MEDIUM')
            colors.append('#f39c12')
        else:
            risk_levels.append('LOW')
            colors.append('#27ae60')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(risks))
    bars = ax.barh(y_pos, impacts, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([risk.replace('_', ' ').title() for risk in risks])
    ax.set_xlabel('Risk Impact Multiplier')
    ax.set_title('Project Risk Analysis', fontsize=16, fontweight='bold')
    
    # Add risk level annotations
    for i, (impact, level) in enumerate(zip(impacts, risk_levels)):
        ax.text(impact + 0.01, i, f'{impact:.3f} ({level})', 
                va='center', fontweight='bold')
    
    # Add vertical line at 1.0 (no impact)
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.7)
    ax.text(1.0, len(risks)-1, 'No Impact', rotation=90, va='top', ha='right')
    
    plt.tight_layout()
    plt.savefig('risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_monte_carlo_distribution(data):
    """Create Monte Carlo cost distribution chart"""
    
    mc_data = data['advanced']['monte_carlo']
    costs = mc_data['simulated_costs']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(costs, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(mc_data['mean_cost'], color='red', linestyle='--', 
                label=f"Mean: ${mc_data['mean_cost']/1e6:.1f}M")
    ax1.axvline(mc_data['median_cost'], color='green', linestyle='--', 
                label=f"Median: ${mc_data['median_cost']/1e6:.1f}M")
    ax1.axvline(mc_data['percentile_90'], color='orange', linestyle='--', 
                label=f"90th %ile: ${mc_data['percentile_90']/1e6:.1f}M")
    
    ax1.set_title('Monte Carlo Cost Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Project Cost (USD)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Box plot
    ax2.boxplot(costs, vert=True)
    ax2.set_title('Cost Distribution Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Project Cost (USD)')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Add percentile annotations
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        value = np.percentile(costs, p)
        ax2.text(1.1, value, f'{p}th: ${value/1e6:.1f}M', va='center')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_tco_breakdown(data):
    """Create Total Cost of Ownership breakdown"""
    
    tco_data = data['system_star']['tco_analysis']
    
    # Pie chart of TCO components
    labels = ['Development', 'Annual Maintenance', 'Enhancements', 'Tech Refresh', 'Scaling']
    sizes = [
        tco_data['development_cost'],
        tco_data['annual_maintenance'] * 5,  # 5 years
        tco_data['enhancement_cycles'],
        tco_data['technology_refresh'],
        tco_data['scaling_costs']
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax1.set_title('5-Year Total Cost of Ownership\n($10.5M)', fontsize=14, fontweight='bold')
    
    # Bar chart showing yearly costs
    years = list(range(1, 6))
    yearly_costs = []
    
    for year in years:
        year_cost = tco_data['annual_maintenance']
        
        # Add enhancement costs every 2 years
        if year % 2 == 0:
            year_cost += tco_data['enhancement_cycles'] / (5 // 2)
        
        # Add tech refresh every 3 years
        if year % 3 == 0:
            year_cost += tco_data['technology_refresh'] / (5 // 3)
        
        # Add scaling costs gradually
        year_cost += tco_data['scaling_costs'] / 5
        
        yearly_costs.append(year_cost)
    
    bars = ax2.bar(years, yearly_costs, color='#3498db')
    ax2.set_title('Annual Operating Costs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Annual Cost (USD)')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    
    # Add value labels
    for bar, cost in zip(bars, yearly_costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f'${cost/1e3:.0f}K', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tco_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sensitivity_analysis_chart(data):
    """Create sensitivity analysis chart"""
    
    sensitivity_data = data['advanced']['sensitivity_analysis']
    drivers = list(sensitivity_data.keys())
    coefficients = [sensitivity_data[driver]['sensitivity_coefficient'] for driver in drivers]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by sensitivity coefficient
    sorted_data = sorted(zip(drivers, coefficients), key=lambda x: x[1], reverse=True)
    drivers_sorted, coefficients_sorted = zip(*sorted_data)
    
    # Create colors based on sensitivity level
    colors = []
    for coef in coefficients_sorted:
        if coef > 0.4:
            colors.append('#e74c3c')  # High sensitivity - red
        elif coef > 0.3:
            colors.append('#f39c12')  # Medium sensitivity - orange
        else:
            colors.append('#27ae60')  # Low sensitivity - green
    
    bars = ax.bar(range(len(drivers_sorted)), coefficients_sorted, color=colors)
    
    ax.set_title('Cost Sensitivity Analysis', fontsize=16, fontweight='bold')
    ax.set_ylabel('Sensitivity Coefficient')
    ax.set_xlabel('Cost Drivers')
    ax.set_xticks(range(len(drivers_sorted)))
    ax.set_xticklabels([driver.replace('_', ' ').title() for driver in drivers_sorted], 
                       rotation=45, ha='right')
    
    # Add value labels
    for bar, coef in zip(bars, coefficients_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{coef:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add sensitivity level legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='High Sensitivity (>0.4)'),
        Patch(facecolor='#f39c12', label='Medium Sensitivity (0.3-0.4)'),
        Patch(facecolor='#27ae60', label='Low Sensitivity (<0.3)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_dashboard():
    """Create a comprehensive dashboard with all key metrics"""
    
    data = load_analysis_data()
    if not data:
        return
    
    print("Creating Cost Estimation Visualizations...")
    print("1. Function Point Analysis...")
    create_function_point_breakdown(data)
    
    print("2. Cost Comparison Chart...")
    create_cost_comparison_chart(data)
    
    print("3. Risk Analysis Chart...")
    create_risk_analysis_chart(data)
    
    print("4. Monte Carlo Distribution...")
    create_monte_carlo_distribution(data)
    
    print("5. TCO Breakdown...")
    create_tco_breakdown(data)
    
    print("6. Sensitivity Analysis...")
    create_sensitivity_analysis_chart(data)
    
    print("\nAll visualizations created successfully!")
    print("Generated files:")
    print("- function_point_analysis.png")
    print("- cost_comparison.png")
    print("- risk_analysis.png")
    print("- monte_carlo_distribution.png")
    print("- tco_breakdown.png")
    print("- sensitivity_analysis.png")

if __name__ == "__main__":
    create_comprehensive_dashboard()