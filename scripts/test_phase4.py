"""
Test script for Phase 4: Environment & Data Enhancement.

Tests Scenario Library, Real-World Data Integration, and Data Augmentation.
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.scenario_library import ScenarioLibrary, ScenarioConfig
from src.data.data_augmentation import (
    DataAugmentationPipeline, AugmentationConfig,
    DistributionType, VehicleType
)
from src.data.real_world_integration import (
    RealWorldDataIntegrator,
    MockTrafficDataProvider, MockWeatherDataProvider, MockEventDataProvider,
    TransferLearningManager
)


def test_scenario_library():
    """Test Phase 4.1: Comprehensive Scenario Library."""
    print("="*80)
    print("Testing Phase 4.1: Comprehensive Scenario Library")
    print("="*80)
    
    library = ScenarioLibrary()
    
    # List all scenarios
    scenarios = library.list_scenarios()
    print(f"\n📊 Available Scenarios: {len(scenarios)}")
    print(f"   Scenarios: {', '.join(scenarios[:10])}...")
    
    # Test temporal scenarios
    print(f"\n📊 Testing Temporal Scenarios...")
    temporal_scenarios = library.get_scenarios_by_type("temporal")
    print(f"   Found {len(temporal_scenarios)} temporal scenarios")
    for scenario in temporal_scenarios[:3]:
        print(f"   - {scenario.name}: {scenario.description}")
    
    # Test weather scenarios
    print(f"\n📊 Testing Weather Scenarios...")
    weather_scenarios = library.get_scenarios_by_type("weather")
    print(f"   Found {len(weather_scenarios)} weather scenarios")
    for scenario in weather_scenarios:
        print(f"   - {scenario.name}: visibility_reduction={scenario.visibility_reduction:.2f}")
    
    # Test event scenarios
    print(f"\n📊 Testing Event Scenarios...")
    event_scenarios = library.get_scenarios_by_type("event")
    print(f"   Found {len(event_scenarios)} event scenarios")
    for scenario in event_scenarios:
        print(f"   - {scenario.name}: severity={scenario.event_severity:.2f}")
    
    # Test applying scenario
    print(f"\n📊 Testing Scenario Application...")
    base_config = {
        "num_lanes": 4,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40
    }
    
    scenario = library.get_scenario("morning_rush")
    modified_config = library.apply_scenario(base_config, scenario)
    print(f"   Base arrival rates: {base_config['arrival_rates']}")
    print(f"   Modified arrival rates: {modified_config['arrival_rates']}")
    print(f"   Traffic multiplier: {scenario.traffic_multiplier}")
    
    # Test random scenario
    random_scenario = library.sample_random_scenario()
    print(f"\n   Random scenario: {random_scenario.name}")
    
    print(f"\n✅ Scenario Library Test Complete!")
    return True


def test_data_augmentation():
    """Test Phase 4.3: Advanced Data Augmentation."""
    print("\n" + "="*80)
    print("Testing Phase 4.3: Advanced Data Augmentation")
    print("="*80)
    
    # Test traffic flow augmentation
    print(f"\n📊 Testing Traffic Flow Augmentation...")
    config = AugmentationConfig(
        flow_distribution=DistributionType.POISSON,
        flow_variance=0.1
    )
    pipeline = DataAugmentationPipeline(config)
    
    base_rates = [0.3, 0.25, 0.35, 0.2]
    augmented_rates = pipeline.flow_aug.augment_arrival_rates(base_rates)
    print(f"   Base rates: {base_rates}")
    print(f"   Augmented rates: {[f'{r:.3f}' for r in augmented_rates]}")
    
    # Test different distributions
    distributions = [DistributionType.POISSON, DistributionType.WEIBULL, 
                    DistributionType.LOG_NORMAL, DistributionType.EXPONENTIAL]
    for dist in distributions:
        config.flow_distribution = dist
        pipeline = DataAugmentationPipeline(config)
        rates = pipeline.flow_aug.augment_arrival_rates([0.3])
        print(f"   {dist.value}: {rates[0]:.3f}")
    
    # Test vehicle type augmentation
    print(f"\n📊 Testing Vehicle Type Augmentation...")
    config.vehicle_type_diversity = True
    pipeline = DataAugmentationPipeline(config)
    
    vehicle_types = []
    for _ in range(10):
        vtype = pipeline.vehicle_aug.sample_vehicle_type()
        vehicle_types.append(vtype)
    
    print(f"   Sampled vehicle types: {vehicle_types}")
    props = pipeline.vehicle_aug.get_vehicle_properties(VehicleType.CAR.value)
    print(f"   Car properties: {props}")
    
    # Test pedestrian augmentation
    print(f"\n📊 Testing Pedestrian Augmentation...")
    config.pedestrian_enabled = True
    config.pedestrian_rate = 0.1
    pipeline = DataAugmentationPipeline(config)
    
    pedestrian_count = 0
    for t in range(100):
        if pipeline.pedestrian_aug.should_pedestrian_cross(t):
            pedestrian_count += 1
    
    print(f"   Pedestrians in 100 steps: {pedestrian_count}")
    
    # Test emergency vehicle augmentation
    print(f"\n📊 Testing Emergency Vehicle Augmentation...")
    config.emergency_enabled = True
    config.emergency_rate = 0.01
    pipeline = DataAugmentationPipeline(config)
    
    emergency_count = 0
    for t in range(3600):  # 1 hour
        if pipeline.emergency_aug.should_emergency_arrive(t):
            emergency_count += 1
    
    print(f"   Emergency vehicles in 1 hour: {emergency_count}")
    
    # Test sensor noise augmentation
    print(f"\n📊 Testing Sensor Noise Augmentation...")
    config.sensor_noise_enabled = True
    config.sensor_noise_level = 0.05
    pipeline = DataAugmentationPipeline(config)
    
    observation = np.array([0.5, 0.3, 0.7, 0.4])
    noisy_observation = pipeline.augment_observation(observation)
    print(f"   Original: {observation}")
    print(f"   Noisy: {noisy_observation}")
    
    # Test complete pipeline
    print(f"\n📊 Testing Complete Pipeline...")
    base_config = {
        "num_lanes": 4,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2]
    }
    augmented_config = pipeline.augment_environment_config(base_config)
    print(f"   Augmented config keys: {list(augmented_config.keys())}")
    
    print(f"\n✅ Data Augmentation Test Complete!")
    return True


def test_real_world_integration():
    """Test Phase 4.2: Real-World Data Integration."""
    print("\n" + "="*80)
    print("Testing Phase 4.2: Real-World Data Integration")
    print("="*80)
    
    # Create mock providers
    traffic_provider = MockTrafficDataProvider(base_flow_rate=300.0)
    weather_provider = MockWeatherDataProvider()
    event_provider = MockEventDataProvider()
    
    integrator = RealWorldDataIntegrator(
        traffic_provider=traffic_provider,
        weather_provider=weather_provider,
        event_provider=event_provider
    )
    
    # Test traffic data
    print(f"\n📊 Testing Traffic Data Provider...")
    location_id = "intersection_001"
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    
    historical_data = traffic_provider.get_historical_data(location_id, start_time, end_time)
    print(f"   Collected {len(historical_data)} data points")
    if historical_data:
        latest = historical_data[-1]
        print(f"   Latest: flow_rate={latest.flow_rate:.1f} vph, "
              f"occupancy={latest.occupancy:.1f}%, speed={latest.speed:.1f} km/h")
    
    # Test weather data
    print(f"\n📊 Testing Weather Data Provider...")
    weather = weather_provider.get_current_weather(location_id)
    if weather:
        print(f"   Condition: {weather.condition}")
        print(f"   Temperature: {weather.temperature:.1f}°C")
        print(f"   Visibility: {weather.visibility:.1f} km")
        print(f"   Precipitation: {weather.precipitation:.1f} mm")
    
    forecast = weather_provider.get_forecast(location_id, days=3)
    print(f"   Forecast: {len(forecast)} days")
    
    # Test event data
    print(f"\n📊 Testing Event Data Provider...")
    events = event_provider.get_upcoming_events(location_id, days=30)
    print(f"   Upcoming events: {len(events)}")
    for event in events[:3]:
        print(f"   - {event.name} ({event.event_type}): {event.start_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Test complete integration
    print(f"\n📊 Testing Complete Integration...")
    context = integrator.get_environment_context(location_id)
    print(f"   Context keys: {list(context.keys())}")
    print(f"   Traffic: {context.get('traffic') is not None}")
    print(f"   Weather: {context.get('weather') is not None}")
    print(f"   Events: {len(context.get('events', []))}")
    
    # Test applying context to config
    base_config = {
        "num_lanes": 4,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40
    }
    modified_config = integrator.apply_context_to_config(base_config, context)
    print(f"   Modified config has real_world_context: {'real_world_context' in modified_config}")
    
    # Test transfer learning
    print(f"\n📊 Testing Transfer Learning Manager...")
    transfer_manager = TransferLearningManager(integrator)
    
    location_ids = ["intersection_001", "intersection_002", "intersection_003"]
    transfer_manager.collect_pretraining_data(location_ids, start_time, end_time)
    pretraining_data = transfer_manager.get_pretraining_dataset()
    print(f"   Pre-training data points: {len(pretraining_data)}")
    
    transfer_manager.collect_finetuning_data("intersection_001", start_time, end_time)
    finetuning_data = transfer_manager.get_finetuning_dataset()
    print(f"   Fine-tuning data points: {len(finetuning_data)}")
    
    print(f"\n✅ Real-World Data Integration Test Complete!")
    return True


def test_combined():
    """Test Phase 4 components together."""
    print("\n" + "="*80)
    print("Testing Phase 4: Combined")
    print("="*80)
    
    # Create all components
    library = ScenarioLibrary()
    config = AugmentationConfig(
        flow_distribution=DistributionType.POISSON,
        vehicle_type_diversity=True,
        pedestrian_enabled=True,
        emergency_enabled=True,
        sensor_noise_enabled=True
    )
    pipeline = DataAugmentationPipeline(config)
    integrator = RealWorldDataIntegrator()
    
    # Get scenario
    scenario = library.get_scenario("rush_hour_rain")
    print(f"\n📊 Scenario: {scenario.name}")
    
    # Get real-world context
    context = integrator.get_environment_context("intersection_001")
    print(f"   Real-world context: traffic={context.get('traffic') is not None}, "
          f"weather={context.get('weather') is not None}")
    
    # Apply scenario
    base_config = {
        "num_lanes": 4,
        "arrival_rates": [0.3, 0.25, 0.35, 0.2],
        "queue_capacity": 40
    }
    scenario_config = library.apply_scenario(base_config, scenario)
    
    # Apply real-world context
    context_config = integrator.apply_context_to_config(scenario_config, context)
    
    # Apply augmentation
    final_config = pipeline.augment_environment_config(context_config)
    
    print(f"   Final config keys: {list(final_config.keys())}")
    print(f"   Arrival rates: {final_config.get('arrival_rates', 'N/A')}")
    
    print(f"\n✅ Combined Test Complete!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("Phase 4 Complete Test Suite")
    print("="*80)
    
    # Test Phase 4.1
    test_scenario_library()
    
    # Test Phase 4.2
    test_real_world_integration()
    
    # Test Phase 4.3
    test_data_augmentation()
    
    # Test Combined
    test_combined()
    
    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

