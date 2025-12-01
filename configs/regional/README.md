# Regional Configuration Examples

This directory contains example configuration files for different regional scenarios. These configurations serve as starting points for adapting the Adaptive Traffic Signal Control System to specific regions worldwide.

## Configuration Files

### 1. `left_hand_drive.json`
**For:** Left-hand drive countries (UK, India, Japan, Australia, etc.)

**Key Features:**
- Standard 4-way intersection configuration
- Notes on phase adjustments for left-hand traffic
- Turning movement considerations

**Usage:** Use as base configuration for left-hand drive regions, then adjust arrival rates and timing based on local traffic patterns.

---

### 2. `mixed_traffic_high_density.json`
**For:** High-density mixed traffic scenarios (Indian cities, Southeast Asia)

**Key Features:**
- Higher queue capacity (60) for mixed vehicle types
- Longer green/yellow/all-red times for safety
- Lower confidence threshold (0.35) for detecting small vehicles
- Configuration for motorcycles, 3-wheelers, and diverse vehicle mix

**Usage:** Ideal for regions with:
- High percentage of 2-wheelers and 3-wheelers
- Poor lane discipline
- Dense, chaotic traffic patterns
- Limited infrastructure

**Recommended Stack:** Fuzzy Logic + YOLOv8-small (edge deployment)

---

### 3. `right_hand_drive_standard.json`
**For:** Standard right-hand drive countries (USA, Canada, Europe, China)

**Key Features:**
- Standard configuration for right-hand traffic
- Good lane discipline assumptions
- Standard vehicle mix (cars, trucks, buses)

**Usage:** Use as base for right-hand drive regions with standard traffic patterns.

**Recommended Stack:** Fuzzy Logic or DQN + YOLOv8-medium (cloud/hybrid deployment)

---

### 4. `low_density_suburban.json`
**For:** Low-density suburban areas

**Key Features:**
- Lower arrival rates (0.12-0.2)
- Lower queue capacity (30)
- Shorter max green time (45s)
- Lower reward weights

**Usage:** Ideal for:
- Suburban intersections
- Lower traffic volumes
- Cost-sensitive deployments
- Simple intersections

**Recommended Stack:** Fuzzy Logic or Webster's + YOLOv8-small/nano (edge deployment)

---

### 5. `high_density_urban_developed.json`
**For:** High-density urban areas in developed regions (New York, London, Tokyo)

**Key Features:**
- High arrival rates (0.45-0.6)
- High queue capacity (60)
- Longer max green time (90s)
- Higher reward weights for congestion reduction
- Support for advanced features (pedestrian phases, bicycle integration, transit priority)

**Usage:** Ideal for:
- Dense urban centers
- Excellent infrastructure
- Multi-intersection coordination
- Advanced feature requirements

**Recommended Stack:** DQN or Fuzzy Logic + YOLOv8-medium/large + LSTM/GNN (cloud/hybrid deployment)

---

### 6. `surat_gujarat_india.json`
**For:** Surat, Gujarat, India - High-density mixed traffic urban city

**Key Features:**
- City-specific configuration for Surat, Gujarat
- High queue capacity (65) for mixed vehicle types
- Longer yellow (4s) and all-red (2s) times for safety
- Lower confidence threshold (0.35) for detecting small vehicles
- Comprehensive vehicle mix (motorcycles 30-40%, auto-rickshaws 15-20%)
- Smart City Mission integration considerations
- Industrial area (Hazira) and commercial zone adaptations

**Usage:** Specifically designed for:
- Surat, Gujarat, India intersections
- High-density mixed traffic with poor lane discipline
- Industrial and commercial zones
- Integration with Surat Smart City infrastructure
- Edge computing deployment

**Recommended Stack:** Fuzzy Logic + YOLOv8-small (edge deployment with cloud backup)

**Special Considerations:**
- Monsoon season adjustments (June-September)
- Festival traffic patterns (Navratri, Diwali)
- Industrial area heavy truck traffic
- BRTS (Bus Rapid Transit System) integration
- Integration with Surat Smart City Command and Control Center

**Deployment Phases:**
1. **Phase 1**: Single intersection pilot (3-6 months)
2. **Phase 2**: 5-10 key intersections (6-12 months)
3. **Phase 3**: City-wide deployment 50+ intersections (12-24 months)

**Cost Estimate:** $15K-$25K per intersection

---

## How to Use These Configurations

### Step 1: Select Base Configuration
Choose the configuration file that best matches your regional characteristics:
- Traffic direction (left/right-hand drive)
- Traffic density (low/medium/high)
- Vehicle mix (standard/mixed)
- Infrastructure level (developed/developing/emerging)

### Step 2: Customize Parameters
Adjust the following based on your specific intersection:
- **arrival_rates**: Measure actual traffic arrival rates per lane
- **queue_capacity**: Based on lane length and observed maximum queues
- **min_green / max_green**: Based on safety requirements and traffic needs
- **reward_weights**: Based on regional priorities (congestion vs wait time)

### Step 3: Vision System Configuration
Configure vision system based on:
- Vehicle types in your region
- Camera quality and positioning
- Detection challenges (small vehicles, overlapping, etc.)
- Confidence thresholds

### Step 4: Regional Adaptations
Review and implement regional-specific adaptations:
- Traffic direction adjustments
- Mixed traffic handling
- Cultural/behavioral factors
- Pedestrian considerations
- Emergency vehicle integration

### Step 5: Testing and Tuning
1. Deploy with base configuration
2. Monitor performance for 1-2 weeks
3. Adjust parameters based on observations
4. Fine-tune reward weights
5. Optimize vision thresholds

---

## Configuration Parameter Guide

### Traffic Pattern Parameters

#### `arrival_rates`
- **Range**: 0.1 to 0.7 (vehicles per second per lane)
- **Low density**: 0.1-0.2
- **Medium density**: 0.2-0.4
- **High density**: 0.4-0.7
- **Measurement**: Count vehicles arriving per lane over 1-hour period, divide by 3600

#### `queue_capacity`
- **Range**: 20 to 80 (vehicles per lane)
- **Low density**: 20-30
- **Medium density**: 30-50
- **High density**: 50-80
- **Calculation**: Based on lane length and average vehicle length

### Signal Timing Parameters

#### `min_green`
- **Range**: 5-15 seconds
- **Purpose**: Safety minimum, pedestrian crossing time
- **Typical**: 5-10 seconds

#### `max_green`
- **Range**: 30-120 seconds
- **Low density**: 30-45 seconds
- **Medium density**: 45-60 seconds
- **High density**: 60-120 seconds

#### `cycle_yellow`
- **Range**: 3-5 seconds
- **Based on**: Approach speed, intersection width
- **Typical**: 3 seconds

#### `cycle_all_red`
- **Range**: 1-3 seconds
- **Purpose**: Clearance time for intersection
- **Typical**: 1-2 seconds

### Reward Function Parameters

#### `queue`
- **Range**: -2.0 to -0.5
- **Higher (more negative)**: Prioritize queue reduction
- **Typical**: -1.0 to -1.5

#### `wait_penalty`
- **Range**: -0.3 to -0.05
- **Higher (more negative)**: Prioritize wait time reduction
- **Typical**: -0.1 to -0.2

#### `efficiency`
- **Range**: 0.01 to 0.05
- **Purpose**: Reward smooth traffic flow
- **Typical**: 0.01 to 0.03

---

## Regional Adaptation Checklist

When adapting a configuration for your region, ensure you've addressed:

- [ ] Traffic direction (left/right-hand drive)
- [ ] Traffic density (measured arrival rates)
- [ ] Vehicle mix (types and percentages)
- [ ] Lane discipline (good/moderate/poor)
- [ ] Infrastructure (power, network, hardware)
- [ ] Signal timing (min/max green, yellow, all-red)
- [ ] Vision system (model size, thresholds, vehicle types)
- [ ] Reward function (priorities and weights)
- [ ] Regional behaviors (aggressive/conservative driving)
- [ ] Special considerations (pedestrians, bicycles, transit, emergency vehicles)

---

## Technology Stack Recommendations by Region Type

### Developed Regions (High Infrastructure)
- **Control**: DQN or Fuzzy Logic
- **Vision**: YOLOv8-medium/large
- **Forecasting**: LSTM/GNN
- **Deployment**: Cloud/Hybrid
- **Cost**: $30K-$50K per intersection

### Developing Regions (Medium Infrastructure)
- **Control**: Fuzzy Logic
- **Vision**: YOLOv8-small
- **Forecasting**: Optional LSTM
- **Deployment**: Edge/Cloud
- **Cost**: $15K-$30K per intersection

### Emerging Regions (Low Infrastructure)
- **Control**: Fuzzy Logic
- **Vision**: YOLOv8-nano/small
- **Forecasting**: None
- **Deployment**: Edge
- **Cost**: $10K-$20K per intersection

---

## Additional Resources

- **Main Checklist**: See `docs/REGIONAL_ADAPTATION_CHECKLIST.md` for comprehensive adaptation guide
- **Deployment Template**: See `docs/case_studies/REGIONAL_DEPLOYMENT_TEMPLATE.md` for documenting deployments
- **Indian Traffic Assessment**: See `HONEST_INDIAN_TRAFFIC_ASSESSMENT.md` for detailed regional analysis example

---

## Contributing

If you create a regional configuration that works well for your region, consider:
1. Documenting it using the Regional Deployment Template
2. Sharing lessons learned
3. Contributing your configuration (with sensitive data removed) to help others

---

**Last Updated**: 2025  
**Maintained By**: Adaptive Traffic Signal Control System Team

