#!/usr/bin/env python3
"""
Comprehensive Cost Estimation Analysis for Adaptive Traffic System
Using Function Point (FP) Analysis and COCOMO Model

Project: Adaptive Traffic Signal Control System
Author: AI Cost Estimation Tool
Date: 2025-10-04
"""

import math
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple


class FunctionPointAnalysis:
    """
    Function Point Analysis for the Adaptive Traffic System
    Based on IFPUG (International Function Point Users Group) standards
    """
    
    def __init__(self):
        # Complexity weights for Function Point calculation
        self.complexity_weights = {
            'simple': {'EI': 3, 'EO': 4, 'EQ': 3, 'ILF': 7, 'EIF': 5},
            'average': {'EI': 4, 'EO': 5, 'EQ': 4, 'ILF': 10, 'EIF': 7},
            'complex': {'EI': 6, 'EO': 7, 'EQ': 6, 'ILF': 15, 'EIF': 10}
        }
        
        # Technical Complexity Factors (TCF)
        self.tcf_factors = {
            'data_communications': 4,  # Real-time video/sensor data
            'distributed_processing': 5,  # Multi-agent coordination
            'performance': 5,  # Real-time response requirements
            'heavily_used_config': 4,  # Multiple traffic scenarios
            'transaction_rate': 5,  # High-frequency signal changes
            'online_data_entry': 3,  # Configuration interfaces
            'end_user_efficiency': 4,  # Operator interfaces
            'online_update': 4,  # Real-time parameter updates
            'complex_processing': 5,  # AI/ML algorithms
            'reusability': 4,  # Modular architecture
            'installation_ease': 3,  # Setup complexity
            'operational_ease': 4,  # Monitoring/maintenance
            'multiple_sites': 5,  # Multi-intersection deployment
            'facilitate_change': 4   # Configuration flexibility
        }
    
    def analyze_system_functions(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze the adaptive traffic system and categorize functions
        """
        functions = {
            # External Inputs (EI) - Data entering the system
            'EI': {
                'simple': 2,    # Basic configuration inputs
                'average': 8,   # Traffic data, sensor inputs, user commands
                'complex': 5    # ML model parameters, complex scenarios
            },
            
            # External Outputs (EO) - Processed data leaving the system
            'EO': {
                'simple': 3,    # Status reports, basic metrics
                'average': 6,   # Signal timing decisions, performance reports
                'complex': 4    # ML predictions, optimization results
            },
            
            # External Inquiries (EQ) - Input-output combinations
            'EQ': {
                'simple': 4,    # System status queries
                'average': 8,   # Traffic state queries, configuration checks
                'complex': 3    # Complex analytics queries
            },
            
            # Internal Logical Files (ILF) - Data maintained by the system
            'ILF': {
                'simple': 2,    # Basic configuration files
                'average': 6,   # Traffic scenarios, model parameters
                'complex': 4    # ML models, historical data structures
            },
            
            # External Interface Files (EIF) - Data from other systems
            'EIF': {
                'simple': 1,    # Basic sensor interfaces
                'average': 4,   # SUMO integration, video feeds
                'complex': 3    # Advanced sensor networks, external APIs
            }
        }
        
        return functions
    
    def calculate_unadjusted_fp(self, functions: Dict[str, Dict[str, int]]) -> Tuple[int, Dict[str, int]]:
        """
        Calculate Unadjusted Function Points (UFP)
        """
        ufp_breakdown = {}
        total_ufp = 0
        
        for function_type, complexity_counts in functions.items():
            type_total = 0
            for complexity, count in complexity_counts.items():
                weight = self.complexity_weights[complexity][function_type]
                points = count * weight
                type_total += points
                print(f"  {function_type} ({complexity}): {count} × {weight} = {points}")
            
            ufp_breakdown[function_type] = type_total
            total_ufp += type_total
            print(f"  {function_type} Total: {type_total}")
        
        return total_ufp, ufp_breakdown
    
    def calculate_tcf(self) -> float:
        """
        Calculate Technical Complexity Factor
        """
        total_influence = sum(self.tcf_factors.values())
        tcf = 0.65 + (0.01 * total_influence)
        return tcf
    
    def calculate_adjusted_fp(self, ufp: int, tcf: float) -> int:
        """
        Calculate Adjusted Function Points
        """
        return int(ufp * tcf)
    
    def perform_analysis(self) -> Dict[str, Any]:
        """
        Perform complete Function Point Analysis
        """
        print("=== FUNCTION POINT ANALYSIS ===")
        print("\n1. Function Identification and Counting:")
        
        functions = self.analyze_system_functions()
        ufp, ufp_breakdown = self.calculate_unadjusted_fp(functions)
        
        print(f"\nUnadjusted Function Points (UFP): {ufp}")
        
        print("\n2. Technical Complexity Factor Calculation:")
        tcf = self.calculate_tcf()
        print(f"Technical Complexity Factor (TCF): {tcf:.3f}")
        
        adjusted_fp = self.calculate_adjusted_fp(ufp, tcf)
        print(f"\n3. Adjusted Function Points: {adjusted_fp}")
        
        return {
            'ufp': ufp,
            'ufp_breakdown': ufp_breakdown,
            'tcf': tcf,
            'adjusted_fp': adjusted_fp,
            'functions': functions
        }


class COCOMOAnalysis:
    """
    COCOMO (Constructive Cost Model) Analysis
    Implements COCOMO II model for effort and schedule estimation
    """
    
    def __init__(self):
        # COCOMO II constants for different project types
        self.cocomo_constants = {
            'organic': {'a': 2.4, 'b': 1.05, 'c': 2.5, 'd': 0.38},
            'semi_detached': {'a': 3.0, 'b': 1.12, 'c': 2.5, 'd': 0.35},
            'embedded': {'a': 3.6, 'b': 1.20, 'c': 2.5, 'd': 0.32}
        }
        
        # Effort Adjustment Factors (EAF) for the project
        self.effort_multipliers = {
            'RELY': 1.15,  # Required software reliability (High)
            'DATA': 1.08,  # Database size (High)
            'CPLX': 1.30,  # Product complexity (Very High - AI/ML)
            'RUSE': 1.07,  # Required reusability (High)
            'DOCU': 1.06,  # Documentation match to life-cycle needs (High)
            'TIME': 1.11,  # Execution time constraint (High)
            'STOR': 1.05,  # Main storage constraint (Nominal)
            'PVOL': 1.04,  # Platform volatility (Low)
            'ACAP': 0.85,  # Analyst capability (Very High)
            'PCAP': 0.88,  # Programmer capability (High)
            'PCON': 1.12,  # Personnel continuity (Low)
            'APEX': 0.95,  # Applications experience (High)
            'PLEX': 0.95,  # Platform experience (High)
            'LTEX': 0.95,  # Language and tool experience (High)
            'TOOL': 0.90,  # Use of software tools (High)
            'SITE': 0.93,  # Multisite development (High)
            'SCED': 1.00,  # Required development schedule (Nominal)
        }
    
    def calculate_kloc_from_fp(self, function_points: int, language: str = 'python') -> float:
        """
        Convert Function Points to KLOC using language-specific conversion
        """
        # Source lines per function point by language
        conversion_factors = {
            'python': 53,      # Lines per FP for Python
            'java': 53,        # Lines per FP for Java
            'c++': 55,         # Lines per FP for C++
            'javascript': 47   # Lines per FP for JavaScript
        }
        
        lines_per_fp = conversion_factors.get(language, 53)
        total_lines = function_points * lines_per_fp
        kloc = total_lines / 1000
        
        return kloc
    
    def classify_project_type(self) -> str:
        """
        Classify the project type based on characteristics
        """
        # The adaptive traffic system is embedded due to:
        # - Real-time constraints
        # - Hardware interface requirements
        # - Safety-critical nature
        # - Complex algorithms (AI/ML)
        return 'embedded'
    
    def calculate_eaf(self) -> float:
        """
        Calculate Effort Adjustment Factor
        """
        eaf = 1.0
        for factor, multiplier in self.effort_multipliers.items():
            eaf *= multiplier
        return eaf
    
    def calculate_effort_and_schedule(self, kloc: float, project_type: str) -> Dict[str, float]:
        """
        Calculate effort (person-months) and schedule (months)
        """
        constants = self.cocomo_constants[project_type]
        eaf = self.calculate_eaf()
        
        # Basic COCOMO effort calculation
        effort_basic = constants['a'] * (kloc ** constants['b'])
        
        # Adjusted effort with EAF
        effort_adjusted = effort_basic * eaf
        
        # Schedule calculation
        schedule = constants['c'] * (effort_adjusted ** constants['d'])
        
        # Team size
        team_size = effort_adjusted / schedule
        
        return {
            'effort_basic': effort_basic,
            'effort_adjusted': effort_adjusted,
            'schedule': schedule,
            'team_size': team_size,
            'eaf': eaf
        }
    
    def perform_analysis(self, function_points: int) -> Dict[str, Any]:
        """
        Perform complete COCOMO analysis
        """
        print("\n=== COCOMO ANALYSIS ===")
        
        # Convert FP to KLOC
        kloc = self.calculate_kloc_from_fp(function_points)
        print(f"\n1. Size Estimation:")
        print(f"   Function Points: {function_points}")
        print(f"   Estimated KLOC (Python): {kloc:.2f}")
        
        # Classify project
        project_type = self.classify_project_type()
        print(f"\n2. Project Classification: {project_type.upper()}")
        
        # Calculate effort and schedule
        results = self.calculate_effort_and_schedule(kloc, project_type)
        
        print(f"\n3. Effort Calculation:")
        print(f"   Basic Effort: {results['effort_basic']:.2f} person-months")
        print(f"   Effort Adjustment Factor (EAF): {results['eaf']:.3f}")
        print(f"   Adjusted Effort: {results['effort_adjusted']:.2f} person-months")
        
        print(f"\n4. Schedule and Team:")
        print(f"   Development Time: {results['schedule']:.2f} months")
        print(f"   Average Team Size: {results['team_size']:.1f} people")
        
        return {
            'kloc': kloc,
            'project_type': project_type,
            **results
        }


class CostEstimator:
    """
    Calculate project costs based on effort estimates
    """
    
    def __init__(self):
        # Average monthly rates by role (USD)
        self.monthly_rates = {
            'senior_developer': 12000,
            'mid_developer': 8000,
            'junior_developer': 5000,
            'ml_engineer': 14000,
            'devops_engineer': 10000,
            'qa_engineer': 7000,
            'project_manager': 11000,
            'architect': 15000
        }
        
        # Team composition percentages
        self.team_composition = {
            'senior_developer': 0.20,
            'mid_developer': 0.30,
            'junior_developer': 0.15,
            'ml_engineer': 0.15,
            'devops_engineer': 0.05,
            'qa_engineer': 0.10,
            'project_manager': 0.03,
            'architect': 0.02
        }
    
    def calculate_development_cost(self, effort_months: float) -> Dict[str, Any]:
        """
        Calculate development costs
        """
        costs_by_role = {}
        total_cost = 0
        
        for role, percentage in self.team_composition.items():
            role_effort = effort_months * percentage
            role_cost = role_effort * self.monthly_rates[role]
            costs_by_role[role] = {
                'effort_months': role_effort,
                'monthly_rate': self.monthly_rates[role],
                'total_cost': role_cost
            }
            total_cost += role_cost
        
        return {
            'costs_by_role': costs_by_role,
            'total_development_cost': total_cost
        }
    
    def calculate_additional_costs(self, development_cost: float) -> Dict[str, float]:
        """
        Calculate additional project costs
        """
        return {
            'infrastructure': development_cost * 0.15,  # 15% for cloud/hardware
            'tools_licenses': development_cost * 0.08,  # 8% for development tools
            'testing_qa': development_cost * 0.20,      # 20% for testing
            'documentation': development_cost * 0.10,   # 10% for documentation
            'training': development_cost * 0.05,        # 5% for training
            'contingency': development_cost * 0.15      # 15% contingency
        }
    
    def perform_cost_analysis(self, effort_months: float) -> Dict[str, Any]:
        """
        Perform complete cost analysis
        """
        print("\n=== COST ESTIMATION ===")
        
        # Development costs
        dev_costs = self.calculate_development_cost(effort_months)
        
        print(f"\n1. Development Costs (Person-months: {effort_months:.2f}):")
        for role, details in dev_costs['costs_by_role'].items():
            print(f"   {role.replace('_', ' ').title()}: "
                  f"{details['effort_months']:.2f} months × "
                  f"${details['monthly_rate']:,} = "
                  f"${details['total_cost']:,.0f}")
        
        total_dev_cost = dev_costs['total_development_cost']
        print(f"\n   Total Development Cost: ${total_dev_cost:,.0f}")
        
        # Additional costs
        additional_costs = self.calculate_additional_costs(total_dev_cost)
        
        print(f"\n2. Additional Costs:")
        total_additional = 0
        for category, cost in additional_costs.items():
            print(f"   {category.replace('_', ' ').title()}: ${cost:,.0f}")
            total_additional += cost
        
        total_project_cost = total_dev_cost + total_additional
        
        print(f"\n3. Total Project Cost: ${total_project_cost:,.0f}")
        
        return {
            'development_costs': dev_costs,
            'additional_costs': additional_costs,
            'total_development_cost': total_dev_cost,
            'total_additional_cost': total_additional,
            'total_project_cost': total_project_cost
        }


def perform_comprehensive_analysis():
    """
    Perform comprehensive cost estimation analysis
    """
    print("=" * 60)
    print("COMPREHENSIVE COST ESTIMATION ANALYSIS")
    print("Adaptive Traffic Signal Control System")
    print("=" * 60)
    
    # Function Point Analysis
    fp_analyzer = FunctionPointAnalysis()
    fp_results = fp_analyzer.perform_analysis()
    
    # COCOMO Analysis
    cocomo_analyzer = COCOMOAnalysis()
    cocomo_results = cocomo_analyzer.perform_analysis(fp_results['adjusted_fp'])
    
    # Cost Analysis
    cost_estimator = CostEstimator()
    cost_results = cost_estimator.perform_cost_analysis(cocomo_results['effort_adjusted'])
    
    # Summary Report
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    
    print(f"\nProject Size:")
    print(f"  • Function Points: {fp_results['adjusted_fp']}")
    print(f"  • Estimated KLOC: {cocomo_results['kloc']:.2f}")
    print(f"  • Actual KLOC: 12.9 (measured)")
    
    print(f"\nEffort Estimation:")
    print(f"  • Development Effort: {cocomo_results['effort_adjusted']:.1f} person-months")
    print(f"  • Schedule: {cocomo_results['schedule']:.1f} months")
    print(f"  • Team Size: {cocomo_results['team_size']:.1f} people")
    
    print(f"\nCost Estimation:")
    print(f"  • Development Cost: ${cost_results['total_development_cost']:,.0f}")
    print(f"  • Total Project Cost: ${cost_results['total_project_cost']:,.0f}")
    
    # Accuracy Assessment
    actual_kloc = 12.943  # From our measurement
    estimated_kloc = cocomo_results['kloc']
    accuracy = (1 - abs(estimated_kloc - actual_kloc) / actual_kloc) * 100
    
    print(f"\nAccuracy Assessment:")
    print(f"  • Estimated KLOC: {estimated_kloc:.2f}")
    print(f"  • Actual KLOC: {actual_kloc:.2f}")
    print(f"  • Estimation Accuracy: {accuracy:.1f}%")
    
    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'function_points': fp_results,
        'cocomo': cocomo_results,
        'costs': cost_results,
        'summary': {
            'function_points': fp_results['adjusted_fp'],
            'estimated_kloc': estimated_kloc,
            'actual_kloc': actual_kloc,
            'accuracy_percentage': accuracy,
            'effort_months': cocomo_results['effort_adjusted'],
            'schedule_months': cocomo_results['schedule'],
            'team_size': cocomo_results['team_size'],
            'total_cost': cost_results['total_project_cost']
        }
    }
    
    with open('cost_estimation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: cost_estimation_results.json")
    
    return results


if __name__ == "__main__":
    results = perform_comprehensive_analysis()