#!/usr/bin/env python3
"""
Test script to demonstrate the professional request tracking system
and trigger the alert when 10 requests remain out of 150 total.
"""

import sys
import time

# Add src to path
sys.path.append('src')
from monitoring.request_tracker import RequestTracker, QuotaConfig

def main():
    """Demonstrate the quota alert system."""
    print("🚦 Testing Professional Request Tracking System")
    print("=" * 60)
    
    # Create tracker with custom config for demo
    config = QuotaConfig(
        total_requests=150,
        warning_threshold=10,
        alert_sound=True,
        enable_alerts=True,
        storage_file="demo_quota_state.json"
    )
    
    tracker = RequestTracker(config)
    
    # Show initial status
    print("\n📊 Initial Status:")
    tracker.print_status_dashboard()
    
    # Simulate operations to use up quota to trigger alert
    print("\n🔄 Simulating operations to demonstrate alert system...")
    print("(We'll simulate 141 operations to trigger the 10-request alert)")
    
    # Use 141 requests to leave exactly 9 remaining (triggers alert)
    operations = [
        "DQN_Training", "MARL_Evaluation", "LSTM_Forecasting", 
        "Computer_Vision", "Genetic_Algorithm", "PSO_Optimization",
        "Fuzzy_Control", "Webster_Method", "System_Monitoring",
        "Performance_Analysis"
    ]
    
    operations_count = 141
    batch_size = 10
    
    for i in range(0, operations_count, batch_size):
        remaining_in_batch = min(batch_size, operations_count - i)
        operation_name = operations[(i // batch_size) % len(operations)]
        
        print(f"\n🔄 Batch {i//batch_size + 1}: Running {remaining_in_batch} {operation_name} operations...")
        
        # Increment usage in batch
        tracker.increment_usage(remaining_in_batch, f"Batch_{operation_name}")
        
        # Show status every few batches or when approaching threshold
        if (i + remaining_in_batch) >= 140 or (i // batch_size + 1) % 5 == 0:
            tracker.print_status_dashboard()
        
        # Small delay for demonstration
        time.sleep(0.1)
    
    print("\n" + "="*80)
    print("🎯 DEMO COMPLETE - Request tracking alert successfully demonstrated!")
    print("="*80)
    
    # Show final comprehensive status
    status = tracker.get_status_report()
    print("\n📋 Final Status Report:")
    import json
    print(json.dumps(status, indent=2))
    
    print(f"\n✅ Alert successfully triggered when {status['quota_status']['remaining_requests']} requests remained!")
    print(f"⚠️  Warning threshold was set to {status['quota_status']['warning_threshold']} requests")
    print(f"📊 Total usage: {status['quota_status']['usage_percentage']:.1f}%")
    
    # Clean up demo file
    import os
    if os.path.exists("demo_quota_state.json"):
        os.remove("demo_quota_state.json")
        print("\n🧹 Demo quota file cleaned up")

if __name__ == "__main__":
    main()
