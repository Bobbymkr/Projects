#!/usr/bin/env python3
"""
Comprehensive Accuracy Assessment for Adaptive Traffic Project
Extreme High-Level Evaluator Analysis
"""

import os
import sys
import numpy as np
import json
import time
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

def analyze_codebase_quality():
    """Analyze codebase quality metrics."""
    print("🔍 CODEBASE QUALITY ANALYSIS")
    print("=" * 60)
    
    # Count source files and lines
    src_files = list(Path('src').rglob('*.py'))
    test_files = list(Path('tests').rglob('*.py'))
    
    total_lines = 0
    for file in src_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                total_lines += len(f.readlines())
        except:
            pass
    
    test_lines = 0
    for file in test_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                test_lines += len(f.readlines())
        except:
            pass
    
    # Calculate metrics
    test_coverage_ratio = test_lines / total_lines if total_lines > 0 else 0
    code_organization_score = min(100, len(src_files) * 2)  # Professional organization
    
    print(f"📁 Source Files: {len(src_files)}")
    print(f"📝 Total Source Lines: {total_lines}")
    print(f"🧪 Test Files: {len(test_files)}")
    print(f"🔬 Test Lines: {test_lines}")
    print(f"📊 Test-to-Code Ratio: {test_coverage_ratio:.2f}")
    print(f"🏗️ Code Organization Score: {code_organization_score:.1f}%")
    
    return {
        "source_files": len(src_files),
        "total_lines": total_lines,
        "test_files": len(test_files),
        "test_lines": test_lines,
        "test_coverage_ratio": test_coverage_ratio,
        "code_organization_score": code_organization_score
    }

def analyze_architectural_accuracy():
    """Analyze architectural accuracy and design quality."""
    print("\n🏛️ ARCHITECTURAL ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Check for key architectural components
    components = {
        "RL Agents": ["src/rl/dqn_agent.py", "src/rl/pytorch_dqn.py"],
        "Environments": ["src/env/traffic_env.py", "src/env/sumo_env.py", "src/env/marl_env.py"],
        "Control Systems": ["src/control/fuzzy_control.py", "src/control/webster_method.py"],
        "Computer Vision": ["src/vision/video_pipeline.py", "src/vision/yolo_queue.py"],
        "Forecasting": ["src/forecast/traffic_forecast.py", "src/forecast/gnn_forecast.py"],
        "Optimization": ["src/optimization/genetic_algo.py", "src/optimization/pso.py"],
        "Utilities": ["src/utils/config.py", "src/utils/metrics.py", "src/utils/health.py"]
    }
    
    component_scores = {}
    total_architecture_score = 0
    
    for component, files in components.items():
        existing_files = [f for f in files if os.path.exists(f)]
        completion_rate = len(existing_files) / len(files)
        component_scores[component] = completion_rate * 100
        total_architecture_score += completion_rate
        
        print(f"🔧 {component}: {completion_rate*100:.1f}% ({len(existing_files)}/{len(files)} files)")
    
    architecture_accuracy = (total_architecture_score / len(components)) * 100
    print(f"\n🎯 Overall Architectural Accuracy: {architecture_accuracy:.1f}%")
    
    return {
        "component_scores": component_scores,
        "architecture_accuracy": architecture_accuracy
    }

def analyze_documentation_accuracy():
    """Analyze documentation quality and accuracy."""
    print("\n📚 DOCUMENTATION ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Check for documentation files
    doc_files = [
        "README.md",
        "PROJECT_EXPLANATION.md",
        "COMPREHENSIVE_PROJECT_REPORT.md",
        "TESTING_STRATEGY.md",
        "SYSTEM_STATUS_DASHBOARD.md"
    ]
    
    existing_docs = [f for f in doc_files if os.path.exists(f)]
    doc_completeness = len(existing_docs) / len(doc_files) * 100
    
    # Analyze README quality
    readme_quality = 0
    if os.path.exists("README.md"):
        with open("README.md", 'r', encoding='utf-8') as f:
            readme_content = f.read()
            readme_quality = min(100, len(readme_content) / 50)  # Quality based on content length
    
    print(f"📖 Documentation Files: {len(existing_docs)}/{len(doc_files)}")
    print(f"📋 Documentation Completeness: {doc_completeness:.1f}%")
    print(f"📄 README Quality Score: {readme_quality:.1f}%")
    
    return {
        "doc_completeness": doc_completeness,
        "readme_quality": readme_quality,
        "total_docs": len(existing_docs)
    }

def analyze_configuration_accuracy():
    """Analyze configuration management accuracy."""
    print("\n⚙️ CONFIGURATION ACCURACY ANALYSIS")
    print("=" * 60)
    
    config_files = list(Path('configs').glob('*.json')) if os.path.exists('configs') else []
    
    # Check for key configuration files
    key_configs = ["intersection.json", "morning_rush.json", "evening_rush.json"]
    config_accuracy = 0
    
    for config in key_configs:
        config_path = f"configs/{config}"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    config_accuracy += 1
                    print(f"✅ {config}: Valid JSON configuration")
            except:
                print(f"❌ {config}: Invalid JSON")
        else:
            print(f"⚠️  {config}: Missing")
    
    config_score = (config_accuracy / len(key_configs)) * 100
    print(f"\n🎯 Configuration Accuracy: {config_score:.1f}%")
    
    return {
        "config_files": len(config_files),
        "config_accuracy": config_score
    }

def analyze_dependency_accuracy():
    """Analyze dependency management accuracy."""
    print("\n📦 DEPENDENCY ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Check for dependency files
    dep_files = ["requirements.txt", "pyproject.toml", "setup.py"]
    existing_deps = [f for f in dep_files if os.path.exists(f)]
    
    dependency_score = len(existing_deps) / len(dep_files) * 100
    
    # Try to import key dependencies
    critical_deps = [
        "numpy", "matplotlib", "tensorflow", "torch", 
        "gymnasium", "stable_baselines3", "ultralytics", "opencv-python"
    ]
    
    importable_deps = 0
    for dep in critical_deps:
        try:
            if dep == "opencv-python":
                __import__("cv2")
            else:
                __import__(dep.replace("-", "_"))
            importable_deps += 1
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
    
    import_accuracy = (importable_deps / len(critical_deps)) * 100
    
    print(f"\n📋 Dependency Files: {len(existing_deps)}/{len(dep_files)}")
    print(f"🎯 Import Accuracy: {import_accuracy:.1f}%")
    
    return {
        "dependency_score": dependency_score,
        "import_accuracy": import_accuracy,
        "importable_deps": importable_deps,
        "total_deps": len(critical_deps)
    }

def calculate_overall_accuracy(assessments):
    """Calculate overall project accuracy score."""
    print("\n🏆 OVERALL ACCURACY ASSESSMENT")
    print("=" * 60)
    
    # Weight different aspects
    weights = {
        "architecture": 0.25,
        "code_quality": 0.20,
        "documentation": 0.15,
        "configuration": 0.15,
        "dependencies": 0.25
    }
    
    scores = {
        "architecture": assessments["architecture"]["architecture_accuracy"],
        "code_quality": min(100, assessments["codebase"]["code_organization_score"] + 
                           assessments["codebase"]["test_coverage_ratio"] * 50),
        "documentation": (assessments["documentation"]["doc_completeness"] + 
                         assessments["documentation"]["readme_quality"]) / 2,
        "configuration": assessments["configuration"]["config_accuracy"],
        "dependencies": assessments["dependencies"]["import_accuracy"]
    }
    
    weighted_score = sum(scores[aspect] * weights[aspect] for aspect in weights)
    
    # Determine grade
    if weighted_score >= 90:
        grade = "A+ (Exceptional)"
    elif weighted_score >= 85:
        grade = "A (Excellent)"
    elif weighted_score >= 80:
        grade = "A- (Very Good)"
    elif weighted_score >= 75:
        grade = "B+ (Good)"
    elif weighted_score >= 70:
        grade = "B (Satisfactory)"
    else:
        grade = "C+ (Needs Improvement)"
    
    print(f"🎯 Architecture Accuracy: {scores['architecture']:.1f}%")
    print(f"💻 Code Quality Score: {scores['code_quality']:.1f}%")
    print(f"📚 Documentation Quality: {scores['documentation']:.1f}%")
    print(f"⚙️ Configuration Accuracy: {scores['configuration']:.1f}%")
    print(f"📦 Dependency Accuracy: {scores['dependencies']:.1f}%")
    print(f"\n🏆 OVERALL ACCURACY: {weighted_score:.1f}%")
    print(f"📊 PROJECT GRADE: {grade}")
    
    return {
        "weighted_score": weighted_score,
        "grade": grade,
        "component_scores": scores
    }

def generate_improvement_recommendations(assessments, overall):
    """Generate specific improvement recommendations."""
    print("\n🔧 IMPROVEMENT RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    if overall["component_scores"]["architecture"] < 85:
        recommendations.append("🏗️ Complete missing architectural components")
    
    if overall["component_scores"]["code_quality"] < 80:
        recommendations.append("💻 Improve test coverage and code organization")
    
    if overall["component_scores"]["documentation"] < 85:
        recommendations.append("📚 Enhance documentation completeness")
    
    if overall["component_scores"]["configuration"] < 90:
        recommendations.append("⚙️ Add missing configuration files")
    
    if overall["component_scores"]["dependencies"] < 95:
        recommendations.append("📦 Resolve dependency installation issues")
    
    if not recommendations:
        print("✅ Project meets high accuracy standards!")
        print("🎯 Consider advanced optimizations and performance tuning")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    return recommendations

def main():
    """Run comprehensive accuracy assessment."""
    print("🚦 ADAPTIVE TRAFFIC PROJECT - EXTREME HIGH-LEVEL ACCURACY ASSESSMENT")
    print("=" * 80)
    print(f"Assessment Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run all assessments
    assessments = {
        "codebase": analyze_codebase_quality(),
        "architecture": analyze_architectural_accuracy(),
        "documentation": analyze_documentation_accuracy(),
        "configuration": analyze_configuration_accuracy(),
        "dependencies": analyze_dependency_accuracy()
    }
    
    # Calculate overall accuracy
    overall = calculate_overall_accuracy(assessments)
    
    # Generate recommendations
    recommendations = generate_improvement_recommendations(assessments, overall)
    
    # Save results
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "assessments": assessments,
        "overall_accuracy": overall,
        "recommendations": recommendations
    }
    
    with open("accuracy_assessment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: accuracy_assessment_results.json")
    
    return results

if __name__ == "__main__":
    main()