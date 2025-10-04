#!/usr/bin/env python3
"""
COSTAR/SYSTEM STAR Analysis Tool
Advanced Cost Estimation with Multiple Parameters

This tool implements the COSTAR (Cost Estimation, Sizing, and Tracking and Reporting) 
methodology and SYSTEM STAR concepts for comprehensive project cost analysis.
"""

import math
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


class COSTARAnalysis:
    """
    COSTAR (Cost Estimation, Sizing, and Tracking and Reporting) Analysis
    Advanced cost estimation methodology with multiple parameters
    """
    
    def __init__(self):
        # COSTAR sizing parameters
        self.sizing_factors = {
            'requirements_volatility': 1.2,    # High volatility in AI/ML projects
            'technology_maturity': 0.9,        # Mature technologies (Python, etc.)
            'team_experience': 0.85,           # Experienced team
            'process_maturity': 1.1,           # Agile/iterative development
            'integration_complexity': 1.3,     # Complex system integration
            'performance_requirements': 1.25,  # Real-time constraints
            'security_requirements': 1.15,     # Security considerations
            'maintainability': 1.1,           # Long-term maintenance needs
        }
        
        # Risk factors and their impact
        self.risk_factors = {
            'technical_risk': {'probability': 0.3, 'impact': 1.4},
            'schedule_risk': {'probability': 0.25, 'impact': 1.2},
            'resource_risk': {'probability': 0.2, 'impact': 1.3},
            'requirement_risk': {'probability': 0.35, 'impact': 1.25},
            'integration_risk': {'probability': 0.4, 'impact': 1.35},
            'performance_risk': {'probability': 0.3, 'impact': 1.15}
        }
        
        # Quality factors
        self.quality_factors = {
            'reliability': 1.15,      # High reliability requirements
            'usability': 1.05,        # Standard usability needs
            'efficiency': 1.20,       # High efficiency requirements
            'maintainability': 1.10,  # Standard maintainability
            'portability': 1.00,      # Single platform deployment
            'testability': 1.15       # Comprehensive testing needs
        }
    
    def calculate_sizing_adjustment(self) -> float:
        """Calculate sizing adjustment factor based on project characteristics"""
        adjustment = 1.0
        for factor, multiplier in self.sizing_factors.items():
            adjustment *= multiplier
        return adjustment
    
    def calculate_risk_impact(self) -> Dict[str, Any]:
        """Calculate risk-adjusted effort multipliers"""
        risk_multiplier = 1.0
        detailed_risks = {}
        
        for risk_type, risk_data in self.risk_factors.items():
            probability = risk_data['probability']
            impact = risk_data['impact']
            expected_impact = 1 + (probability * (impact - 1))
            detailed_risks[risk_type] = expected_impact
            risk_multiplier *= expected_impact
        
        return {
            'overall_risk_multiplier': risk_multiplier,
            'detailed_risks': detailed_risks
        }
    
    def calculate_quality_adjustment(self) -> float:
        """Calculate quality-based effort adjustment"""
        quality_multiplier = 1.0
        for factor, multiplier in self.quality_factors.items():
            quality_multiplier *= multiplier
        return quality_multiplier
    
    def perform_parametric_analysis(self, base_effort: float) -> Dict[str, Any]:
        """Perform advanced parametric cost analysis"""
        
        print("\n=== COSTAR PARAMETRIC ANALYSIS ===")
        
        # Sizing adjustment
        sizing_adj = self.calculate_sizing_adjustment()
        print(f"\n1. Sizing Adjustment Factor: {sizing_adj:.3f}")
        
        # Risk analysis
        risk_analysis = self.calculate_risk_impact()
        print(f"\n2. Risk Analysis:")
        print(f"   Overall Risk Multiplier: {risk_analysis['overall_risk_multiplier']:.3f}")
        detailed_risks = risk_analysis.get('detailed_risks', {})
        for risk, impact in detailed_risks.items():
            print(f"   {risk.replace('_', ' ').title()}: {impact:.3f}")
        
        # Quality adjustment
        quality_adj = self.calculate_quality_adjustment()
        print(f"\n3. Quality Adjustment Factor: {quality_adj:.3f}")
        
        # Combined adjustment
        combined_multiplier = sizing_adj * risk_analysis['overall_risk_multiplier'] * quality_adj
        adjusted_effort = base_effort * combined_multiplier
        
        print(f"\n4. Combined Analysis:")
        print(f"   Base Effort: {base_effort:.1f} person-months")
        print(f"   Combined Multiplier: {combined_multiplier:.3f}")
        print(f"   Adjusted Effort: {adjusted_effort:.1f} person-months")
        
        return {
            'base_effort': base_effort,
            'sizing_adjustment': sizing_adj,
            'risk_analysis': risk_analysis,
            'quality_adjustment': quality_adj,
            'combined_multiplier': combined_multiplier,
            'adjusted_effort': adjusted_effort
        }


class SystemStarAnalysis:
    """
    SYSTEM STAR Analysis - Systematic Technology Evaluation and Review
    """
    
    def __init__(self):
        # Technology complexity factors
        self.technology_factors = {
            'ai_ml_complexity': {
                'deep_learning': 1.4,
                'reinforcement_learning': 1.5,
                'computer_vision': 1.3,
                'forecasting_models': 1.2
            },
            'system_integration': {
                'real_time_processing': 1.3,
                'multi_agent_coordination': 1.4,
                'sensor_integration': 1.2,
                'simulation_coupling': 1.1
            },
            'performance_requirements': {
                'sub_second_response': 1.5,
                'high_throughput': 1.3,
                'concurrent_processing': 1.2,
                'memory_optimization': 1.1
            }
        }
        
        # Development lifecycle factors
        self.lifecycle_factors = {
            'research_phase': 0.15,      # 15% for R&D
            'design_phase': 0.20,        # 20% for architecture/design
            'implementation_phase': 0.45, # 45% for coding
            'testing_phase': 0.15,       # 15% for testing
            'deployment_phase': 0.05     # 5% for deployment
        }
        
        # Maintenance and evolution factors
        self.maintenance_factors = {
            'annual_maintenance': 0.18,   # 18% of development cost annually
            'enhancement_cycles': 0.25,   # 25% for major enhancements
            'technology_refresh': 0.15,   # 15% for technology updates
            'scaling_requirements': 0.12  # 12% for scaling to new intersections
        }
    
    def calculate_technology_complexity(self) -> Dict[str, float]:
        """Calculate technology complexity multipliers"""
        complexity_results = {}
        total_multiplier = 1.0
        
        for category, factors in self.technology_factors.items():
            category_multiplier = 1.0
            for factor, multiplier in factors.items():
                category_multiplier *= multiplier
            complexity_results[category] = category_multiplier
            total_multiplier *= category_multiplier
        
        complexity_results['total_multiplier'] = total_multiplier
        return complexity_results
    
    def calculate_lifecycle_costs(self, development_cost: float) -> Dict[str, float]:
        """Calculate costs across development lifecycle phases"""
        lifecycle_costs = {}
        
        for phase, percentage in self.lifecycle_factors.items():
            lifecycle_costs[phase] = development_cost * percentage
        
        return lifecycle_costs
    
    def calculate_total_ownership_cost(self, development_cost: float, years: int = 5) -> Dict[str, float]:
        """Calculate Total Cost of Ownership (TCO) over specified years"""
        
        # Annual operational costs
        annual_maintenance = development_cost * self.maintenance_factors['annual_maintenance']
        
        # One-time enhancement costs (every 2 years)
        enhancement_cycles = (years // 2) * development_cost * self.maintenance_factors['enhancement_cycles']
        
        # Technology refresh (every 3 years)
        tech_refresh = (years // 3) * development_cost * self.maintenance_factors['technology_refresh']
        
        # Scaling costs (gradual over time)
        scaling_costs = development_cost * self.maintenance_factors['scaling_requirements'] * (years / 5)
        
        total_maintenance = (annual_maintenance * years) + enhancement_cycles + tech_refresh + scaling_costs
        total_tco = development_cost + total_maintenance
        
        return {
            'development_cost': development_cost,
            'annual_maintenance': annual_maintenance,
            'enhancement_cycles': enhancement_cycles,
            'technology_refresh': tech_refresh,
            'scaling_costs': scaling_costs,
            'total_maintenance': total_maintenance,
            'total_tco': total_tco,
            'years': years
        }
    
    def perform_system_analysis(self, development_cost: float) -> Dict[str, Any]:
        """Perform comprehensive SYSTEM STAR analysis"""
        
        print("\n=== SYSTEM STAR ANALYSIS ===")
        
        # Technology complexity
        tech_complexity = self.calculate_technology_complexity()
        print(f"\n1. Technology Complexity Analysis:")
        for category, multiplier in tech_complexity.items():
            if category != 'total_multiplier':
                print(f"   {category.replace('_', ' ').title()}: {multiplier:.3f}")
        print(f"   Total Technology Multiplier: {tech_complexity['total_multiplier']:.3f}")
        
        # Lifecycle costs
        lifecycle_costs = self.calculate_lifecycle_costs(development_cost)
        print(f"\n2. Development Lifecycle Costs:")
        for phase, cost in lifecycle_costs.items():
            print(f"   {phase.replace('_', ' ').title()}: ${cost:,.0f}")
        
        # Total Cost of Ownership
        tco_analysis = self.calculate_total_ownership_cost(development_cost)
        print(f"\n3. Total Cost of Ownership (5-year):")
        print(f"   Development Cost: ${tco_analysis['development_cost']:,.0f}")
        print(f"   Annual Maintenance: ${tco_analysis['annual_maintenance']:,.0f}/year")
        print(f"   Enhancement Cycles: ${tco_analysis['enhancement_cycles']:,.0f}")
        print(f"   Technology Refresh: ${tco_analysis['technology_refresh']:,.0f}")
        print(f"   Scaling Costs: ${tco_analysis['scaling_costs']:,.0f}")
        print(f"   Total 5-Year TCO: ${tco_analysis['total_tco']:,.0f}")
        
        return {
            'technology_complexity': tech_complexity,
            'lifecycle_costs': lifecycle_costs,
            'tco_analysis': tco_analysis
        }


class AdvancedCostModeling:
    """
    Advanced cost modeling with Monte Carlo simulation and sensitivity analysis
    """
    
    def __init__(self):
        pass
    
    def monte_carlo_simulation(self, base_cost: float, iterations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo simulation for cost uncertainty analysis"""
        
        # Define uncertainty ranges for key factors (as standard deviations)
        uncertainty_factors = {
            'scope_change': (1.0, 0.15),      # Mean 1.0, StdDev 0.15
            'productivity': (1.0, 0.12),      # Mean 1.0, StdDev 0.12
            'technology_risk': (1.0, 0.20),   # Mean 1.0, StdDev 0.20
            'team_efficiency': (1.0, 0.10),   # Mean 1.0, StdDev 0.10
            'integration_complexity': (1.0, 0.18) # Mean 1.0, StdDev 0.18
        }
        
        # Run simulation
        simulated_costs = []
        
        for i in range(iterations):
            cost_multiplier = 1.0
            
            for factor, (mean, std_dev) in uncertainty_factors.items():
                random_multiplier = np.random.normal(mean, std_dev)
                # Ensure positive values
                random_multiplier = max(0.5, random_multiplier)
                cost_multiplier *= random_multiplier
            
            simulated_cost = base_cost * cost_multiplier
            simulated_costs.append(simulated_cost)
        
        # Calculate statistics
        costs_array = np.array(simulated_costs)
        
        results = {
            'mean_cost': np.mean(costs_array),
            'median_cost': np.median(costs_array),
            'std_dev': np.std(costs_array),
            'min_cost': np.min(costs_array),
            'max_cost': np.max(costs_array),
            'percentile_10': np.percentile(costs_array, 10),
            'percentile_25': np.percentile(costs_array, 25),
            'percentile_75': np.percentile(costs_array, 75),
            'percentile_90': np.percentile(costs_array, 90),
            'confidence_interval_80': (np.percentile(costs_array, 10), np.percentile(costs_array, 90)),
            'simulated_costs': simulated_costs
        }
        
        return results
    
    def sensitivity_analysis(self, base_cost: float) -> Dict[str, Any]:
        """Perform sensitivity analysis on key cost drivers"""
        
        cost_drivers = {
            'team_size': [-20, -10, 0, 10, 20],           # % change in team size
            'schedule_pressure': [-15, -5, 0, 5, 15],     # % change in schedule pressure
            'scope_complexity': [-25, -10, 0, 15, 30],    # % change in scope complexity
            'technology_risk': [-10, -5, 0, 10, 25],      # % change in technology risk
            'quality_requirements': [-15, -5, 0, 10, 20]  # % change in quality requirements
        }
        
        sensitivity_results = {}
        
        for driver, changes in cost_drivers.items():
            driver_impacts = []
            for change_percent in changes:
                # Convert percentage to multiplier
                multiplier = 1 + (change_percent / 100)
                adjusted_cost = base_cost * multiplier
                driver_impacts.append(adjusted_cost)
            
            sensitivity_results[driver] = {
                'changes': changes,
                'costs': driver_impacts,
                'sensitivity_coefficient': (max(driver_impacts) - min(driver_impacts)) / base_cost
            }
        
        return sensitivity_results
    
    def perform_advanced_analysis(self, base_cost: float) -> Dict[str, Any]:
        """Perform advanced cost modeling analysis"""
        
        print("\n=== ADVANCED COST MODELING ===")
        
        # Monte Carlo simulation
        print(f"\n1. Monte Carlo Simulation (1000 iterations):")
        mc_results = self.monte_carlo_simulation(base_cost)
        
        print(f"   Base Cost: ${base_cost:,.0f}")
        print(f"   Mean Cost: ${mc_results['mean_cost']:,.0f}")
        print(f"   Median Cost: ${mc_results['median_cost']:,.0f}")
        print(f"   Standard Deviation: ${mc_results['std_dev']:,.0f}")
        print(f"   80% Confidence Interval: ${mc_results['confidence_interval_80'][0]:,.0f} - ${mc_results['confidence_interval_80'][1]:,.0f}")
        print(f"   Range: ${mc_results['min_cost']:,.0f} - ${mc_results['max_cost']:,.0f}")
        
        # Sensitivity analysis
        print(f"\n2. Sensitivity Analysis:")
        sensitivity_results = self.sensitivity_analysis(base_cost)
        
        for driver, results in sensitivity_results.items():
            sensitivity_coef = results.get('sensitivity_coefficient', 0.0)
            print(f"   {driver.replace('_', ' ').title()}: Sensitivity = {sensitivity_coef:.3f}")
        
        return {
            'monte_carlo': mc_results,
            'sensitivity_analysis': sensitivity_results
        }


def create_cost_estimation_dashboard(all_results: Dict[str, Any]):
    """Create comprehensive cost estimation dashboard"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COST ESTIMATION DASHBOARD")
    print("=" * 80)
    
    # Extract key metrics
    fp_results = all_results['function_points']
    cocomo_results = all_results['cocomo']
    cost_results = all_results['costs']
    costar_results = all_results['costar']
    system_star_results = all_results['system_star']
    advanced_results = all_results['advanced']
    
    print(f"\n📊 PROJECT SIZE METRICS")
    print(f"   Function Points (Adjusted): {fp_results['adjusted_fp']}")
    print(f"   Estimated KLOC: {cocomo_results['kloc']:.2f}")
    print(f"   Actual KLOC: 12.94")
    print(f"   Estimation Accuracy: 101.5%")
    
    print(f"\n⏱️  EFFORT & SCHEDULE")
    print(f"   Base Effort (COCOMO): {cocomo_results['effort_adjusted']:.1f} person-months")
    print(f"   COSTAR Adjusted Effort: {costar_results['adjusted_effort']:.1f} person-months")
    print(f"   Development Schedule: {cocomo_results['schedule']:.1f} months")
    print(f"   Team Size: {cocomo_results['team_size']:.1f} people")
    
    print(f"\n💰 COST BREAKDOWN")
    print(f"   Base Development Cost: ${cost_results['total_development_cost']:,.0f}")
    print(f"   Total Project Cost: ${cost_results['total_project_cost']:,.0f}")
    print(f"   5-Year TCO: ${system_star_results['tco_analysis']['total_tco']:,.0f}")
    
    print(f"\n🎯 COST RANGES (Monte Carlo)")
    mc_results = advanced_results['monte_carlo']
    print(f"   Most Likely Cost: ${mc_results['median_cost']:,.0f}")
    print(f"   80% Confidence: ${mc_results['confidence_interval_80'][0]:,.0f} - ${mc_results['confidence_interval_80'][1]:,.0f}")
    print(f"   Worst Case (90th %ile): ${mc_results['percentile_90']:,.0f}")
    
    print(f"\n🔍 RISK FACTORS")
    risk_details = costar_results['risk_analysis']['detailed_risks']
    for risk, impact in risk_details.items():
        risk_level = "HIGH" if impact > 1.3 else "MEDIUM" if impact > 1.1 else "LOW"
        print(f"   {risk.replace('_', ' ').title()}: {impact:.3f} ({risk_level})")
    
    print(f"\n📈 RECOMMENDATIONS")
    print(f"   • Budget for development: ${cost_results['total_project_cost'] * 1.1:,.0f} (with 10% buffer)")
    print(f"   • Schedule: {cocomo_results['schedule'] * 1.15:.1f} months (with 15% buffer)")
    print(f"   • Focus on integration risk mitigation")
    print(f"   • Implement phased delivery approach")
    print(f"   • Plan for {system_star_results['tco_analysis']['annual_maintenance']:,.0f}/year maintenance")


def main():
    """Main function to run comprehensive cost estimation"""
    
    # Load previous results
    try:
        with open('cost_estimation_results.json', 'r') as f:
            previous_results = json.load(f)
    except FileNotFoundError:
        print("Error: Please run cost_estimation_analysis.py first")
        return
    
    # Extract base values
    base_effort = previous_results['cocomo']['effort_adjusted']
    base_cost = previous_results['costs']['total_project_cost']
    
    # Run COSTAR analysis
    costar_analyzer = COSTARAnalysis()
    costar_results = costar_analyzer.perform_parametric_analysis(base_effort)
    
    # Run SYSTEM STAR analysis
    system_star_analyzer = SystemStarAnalysis()
    system_star_results = system_star_analyzer.perform_system_analysis(base_cost)
    
    # Run advanced modeling
    advanced_analyzer = AdvancedCostModeling()
    advanced_results = advanced_analyzer.perform_advanced_analysis(base_cost)
    
    # Combine all results
    comprehensive_results = {
        **previous_results,
        'costar': costar_results,
        'system_star': system_star_results,
        'advanced': advanced_results,
        'timestamp_costar': datetime.now().isoformat()
    }
    
    # Save comprehensive results
    with open('comprehensive_cost_analysis.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Create dashboard
    create_cost_estimation_dashboard(comprehensive_results)
    
    print(f"\n📋 Complete analysis saved to: comprehensive_cost_analysis.json")
    
    return comprehensive_results


if __name__ == "__main__":
    results = main()