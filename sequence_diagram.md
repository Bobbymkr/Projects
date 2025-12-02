# Sequence Diagram - Adaptive Traffic Signal Control System

This document provides comprehensive Sequence Diagrams for the Adaptive Traffic Signal Control System, illustrating the temporal interactions between components during signal timing decisions, including real-time processing, training workflows, and error handling scenarios.

## Overview

The signal timing decision process involves complex temporal interactions between multiple components: cameras, computer vision processing, queue estimation, DQN agents, environments, signal controllers, and monitoring systems. The sequence diagrams show how these components coordinate during different operational modes.

## Main Signal Timing Decision Sequence

```mermaid
sequenceDiagram
    participant CAM as Camera System
    participant VP as Video Pipeline
    participant CV as Computer Vision
    participant QE as Queue Estimator
    participant AG as Control Agent
    participant ENV as Environment
    participant SC as Signal Controller
    participant LOG as Performance Logger
    participant MON as Monitor

    Note over CAM, MON: Continuous Operation Cycle (Every 100ms)
    
    loop Continuous Video Processing
        CAM->>VP: Video Frame (30 FPS)
        VP->>CV: Processed Frame
        CV->>QE: Vehicle Detections
        QE->>MON: Queue Metrics
    end
    
    Note over CAM, MON: Decision Point (Every Phase Transition)
    
    rect rgb(255, 245, 238)
        Note over QE, ENV: Signal Timing Decision Process
        QE->>ENV: Current Queue State [q1, q2, q3, q4]
        ENV->>AG: Normalized State Vector
        
        alt Model-Based RL (converged)
            AG->>AG: MPC Planning (horizon/candidates reduced)
            AG->>SC: Green Duration Command
        else Hierarchical RL
            AG->>AG: High-Level Phase Selection
            AG->>AG: Low-Level Timing Control
            AG->>SC: Signal Command (Phase + Duration)
        else Classical Controllers (Fuzzy/Webster)
            AG->>AG: Rule-based/Analytical Computation
            AG->>SC: Signal Command (Phase + Duration)
        end
        
        SC->>ENV: Status Confirmation
        ENV->>AG: Reward Signal
        AG->>LOG: Performance Metrics
    end
    
    Note over SC, MON: Signal Execution Phase
    
    loop Green Phase Execution
        SC->>SC: Maintain Green Signal
        QE->>MON: Real-time Metrics
        MON->>LOG: System Status
    end
    
    Note over ENV, SC: Phase Transition
    ENV->>SC: Yellow Phase Command
    SC->>SC: Execute Yellow (3s)
    ENV->>SC: All-Red Phase Command
    SC->>SC: Execute All-Red (1s)
    ENV->>ENV: Switch to Next Phase
```

## Training Mode Sequence Diagram

```mermaid
sequenceDiagram
    participant ENV as Training Environment
    participant DQN as DQN Agent
    participant RB as Replay Buffer
    participant TN as Target Network
    participant OPT as Optimizer
    participant EVAL as Evaluator
    participant SAVE as Model Saver
    participant TB as TensorBoard

    Note over ENV, TB: Training Episode Initialization
    ENV->>ENV: Reset Environment
    ENV->>DQN: Initial State Vector
    
    loop Training Step (Until Episode End)
        DQN->>DQN: State Processing
        DQN->>DQN: ε-Greedy Action Selection
        DQN->>ENV: Selected Action
        
        ENV->>ENV: Environment Step
        ENV->>ENV: Reward Calculation
        ENV->>DQN: Next State + Reward + Done
        
        DQN->>RB: Store Experience (s, a, r, s', done)
        
        alt Training Condition Met
            DQN->>RB: Sample Batch
            RB->>DQN: Training Batch
            DQN->>DQN: Compute Q-targets
            TN->>DQN: Target Q-values
            DQN->>OPT: Compute Loss & Gradients
            OPT->>DQN: Update Parameters
            DQN->>TB: Training Metrics
        end
        
        alt Target Network Update
            DQN->>TN: Copy Parameters
            TN->>TB: Update Confirmation
        end
        
        alt Evaluation Episode
            DQN->>EVAL: Current Policy
            EVAL->>ENV: Evaluation Run
            ENV->>EVAL: Performance Results
            EVAL->>TB: Evaluation Metrics
        end
        
        alt Save Checkpoint
            DQN->>SAVE: Model Parameters
            SAVE->>SAVE: Save to Disk
            SAVE->>TB: Save Confirmation
        end
    end
    
    Note over ENV, TB: Episode Completion
    ENV->>TB: Episode Statistics
    DQN->>TB: Learning Progress
```

## Video-Based Real-Time Decision Sequence

```mermaid
sequenceDiagram
    participant CAM as IP Camera
    participant VIS as Video Input Stream
    participant ROI as ROI Manager
    participant YOLO as YOLO Detector
    participant TRACK as Object Tracker
    participant QE as Queue Estimator
    participant VE as Video Environment
    participant AG as Control Agent
    participant HC as Hardware Controller
    participant WS as WebSocket Server
    participant DASH as Dashboard

    Note over CAM, DASH: Real-Time Video Processing Pipeline
    
    loop Continuous Frame Processing (30 FPS)
        CAM->>VIS: RTSP Video Stream
        VIS->>VIS: Frame Capture & Buffering
        VIS->>ROI: Raw Frame
        ROI->>ROI: Apply Lane ROIs
        ROI->>YOLO: ROI Regions
        
        YOLO->>YOLO: Vehicle Detection
        YOLO->>TRACK: Detection Results
        TRACK->>TRACK: Update Vehicle Tracks
        TRACK->>QE: Tracked Objects
        
        QE->>QE: Calculate Queue Lengths
        QE->>QE: Temporal Smoothing
        QE->>VE: Traffic State Vector
    end
    
    Note over VE, DASH: Decision Making at Phase Boundaries
    
    rect rgb(240, 248, 255)
        Note over VE, HC: Signal Control Decision
        VE->>VE: State Normalization
        VE->>AG: Normalized Observations
        
        alt Model-Based RL (converged)
            AG->>AG: MPC Planning (reduced horizon/candidates)
            AG->>HC: Signal Control Message
        else Hierarchical RL
            AG->>AG: High-Level Phase Selection
            AG->>AG: Low-Level Timing Control
            AG->>HC: Signal Control Message
        else Classical Controllers
            AG->>AG: Fuzzy/Webster Computation
            AG->>HC: Signal Control Message
        end
        
        HC->>VE: Acknowledgment
        VE->>AG: Performance Feedback
    end
    
    Note over WS, DASH: Real-Time Monitoring
    
    par Real-Time Updates
        QE->>WS: Queue Statistics
        AG->>WS: Agent Decisions
        VE->>WS: Environment State
        HC->>WS: Signal Status
    and
        WS->>DASH: Live Data Stream
        DASH->>DASH: Update Visualizations
    end
```

## Multi-Agent Coordination Sequence

```mermaid
sequenceDiagram
    participant ENV as MARL Environment
    participant AG1 as Agent 1 (North)
    participant AG2 as Agent 2 (South)
    participant AG3 as Agent 3 (East)
    participant AG4 as Agent 4 (West)
    participant COORD as Coordinator
    participant FC as Forecaster
    participant SC as Signal Controller

    Note over ENV, SC: Multi-Intersection Coordination
    
    ENV->>ENV: Initialize Multi-Agent Episode
    
    loop Synchronized Decision Making
        ENV->>AG1: Local State + Global Context
        ENV->>AG2: Local State + Global Context
        ENV->>AG3: Local State + Global Context
        ENV->>AG4: Local State + Global Context
        
        par Agent Processing
            AG1->>FC: Request Traffic Forecast
            FC->>AG1: Predicted Traffic Flow
            AG1->>AG1: Local Decision Making
        and
            AG2->>FC: Request Traffic Forecast
            FC->>AG2: Predicted Traffic Flow
            AG2->>AG2: Local Decision Making
        and
            AG3->>FC: Request Traffic Forecast
            FC->>AG3: Predicted Traffic Flow
            AG3->>AG3: Local Decision Making
        and
            AG4->>FC: Request Traffic Forecast
            FC->>AG4: Predicted Traffic Flow
            AG4->>AG4: Local Decision Making
        end
        
        AG1->>COORD: Local Action + Confidence
        AG2->>COORD: Local Action + Confidence
        AG3->>COORD: Local Action + Confidence
        AG4->>COORD: Local Action + Confidence
        
        COORD->>COORD: Conflict Resolution
        COORD->>COORD: Global Optimization
        
        COORD->>AG1: Coordinated Action
        COORD->>AG2: Coordinated Action
        COORD->>AG3: Coordinated Action
        COORD->>AG4: Coordinated Action
        
        par Signal Execution
            AG1->>SC: Signal Command 1
            AG2->>SC: Signal Command 2
            AG3->>SC: Signal Command 3
            AG4->>SC: Signal Command 4
        end
        
        SC->>SC: Execute Coordinated Signals
        SC->>ENV: Global Status Update
        
        ENV->>ENV: Calculate Global Reward
        ENV->>AG1: Local + Global Reward
        ENV->>AG2: Local + Global Reward
        ENV->>AG3: Local + Global Reward
        ENV->>AG4: Local + Global Reward
    end
```

## Error Handling and Recovery Sequence

```mermaid
sequenceDiagram
    participant CAM as Camera
    participant VP as Video Pipeline
    participant DQN as DQN Agent
    participant ENV as Environment
    participant SC as Signal Controller
    participant ERR as Error Handler
    participant FB as Fallback System
    participant LOG as Error Logger

    Note over CAM, LOG: Normal Operation with Error Detection
    
    CAM->>VP: Video Stream
    VP->>VP: Frame Processing
    
    alt Camera Failure
        CAM-xVP: Connection Lost
        VP->>ERR: Camera Error
        ERR->>FB: Activate Fallback Mode
        FB->>ENV: Use Simulation Data
        ERR->>LOG: Log Camera Failure
    end
    
    VP->>DQN: State Data
    DQN->>DQN: Process State
    
    alt Model Inference Error
        DQN-xDQN: Model Exception
        DQN->>ERR: Inference Error
        ERR->>FB: Use Default Policy
        FB->>ENV: Fixed Timing Action
        ERR->>LOG: Log Model Error
    end
    
    DQN->>ENV: Action Decision
    ENV->>SC: Signal Command
    
    alt Hardware Communication Error
        ENV-xSC: Communication Timeout
        ENV->>ERR: Hardware Error
        ERR->>FB: Emergency All-Red
        FB->>SC: Safety Command
        ERR->>LOG: Log Hardware Error
        
        loop Recovery Attempts
            ERR->>SC: Test Connection
            alt Connection Restored
                SC->>ERR: Connection OK
                ERR->>ENV: Resume Normal Operation
            else Connection Failed
                ERR->>ERR: Wait & Retry
            end
        end
    end
    
    Note over ERR, LOG: Error Recovery Procedures
    
    alt System Recovery
        ERR->>ERR: Assess System State
        ERR->>FB: Determine Recovery Strategy
        
        par Recovery Actions
            FB->>CAM: Restart Camera Connection
            FB->>DQN: Reload Model
            FB->>SC: Test Hardware Interface
        end
        
        FB->>ERR: Recovery Status
        ERR->>LOG: Recovery Results
        
        alt Recovery Successful
            ERR->>ENV: Resume Normal Operation
            ENV->>DQN: Continue Processing
        else Recovery Failed
            ERR->>FB: Maintain Safe Mode
            FB->>SC: Fixed Timing Pattern
        end
    end
```

## Performance Monitoring Sequence

```mermaid
sequenceDiagram
    participant SYS as System Components
    participant PM as Performance Monitor
    participant MET as Metrics Collector
    participant TS as Time Series DB
    participant ALERT as Alert Manager
    participant DASH as Dashboard
    participant ADMIN as Administrator

    Note over SYS, ADMIN: Continuous Performance Monitoring
    
    loop Every 1 Second
        SYS->>PM: System Metrics
        PM->>MET: Collect Performance Data
        
        par Metric Collection
            MET->>MET: CPU Usage
            MET->>MET: Memory Usage
            MET->>MET: Network I/O
            MET->>MET: Frame Processing Rate
            MET->>MET: Decision Latency
            MET->>MET: Queue Detection Accuracy
        end
        
        MET->>TS: Store Metrics
        MET->>PM: Current Values
        
        alt Performance Threshold Exceeded
            PM->>ALERT: Threshold Violation
            ALERT->>ALERT: Evaluate Severity
            
            alt Critical Alert
                ALERT->>ADMIN: Immediate Notification
                ADMIN->>SYS: Manual Intervention
            else Warning Alert
                ALERT->>DASH: Dashboard Warning
                DASH->>DASH: Display Warning
            end
        end
    end
    
    Note over TS, DASH: Real-Time Dashboard Updates
    
    loop Every 5 Seconds
        TS->>DASH: Historical Metrics
        DASH->>DASH: Update Charts
        DASH->>DASH: Calculate Trends
        DASH->>DASH: Refresh Display
    end
    
    Note over ADMIN, SYS: Administrative Actions
    
    alt Performance Optimization
        ADMIN->>SYS: Adjust Configuration
        SYS->>PM: New Performance Profile
        PM->>TS: Updated Baseline
    end
```

## Timing Analysis and Performance Requirements

### Real-Time Constraints

| **Component** | **Processing Time** | **Frequency** | **Latency Requirement** |
|---------------|-------------------|---------------|------------------------|
| **Frame Capture** | 33ms | 30 FPS | < 50ms |
| **YOLO Detection** | 50ms | 20 FPS | < 100ms |
| **Queue Estimation** | 10ms | 30 FPS | < 20ms |
| **DQN Inference** | 5ms | On-demand | < 10ms |
| **Signal Command** | 2ms | On-demand | < 5ms |
| **Total Decision** | ~100ms | Per phase | < 200ms |

### Sequence Timing Patterns

```yaml
# Timing Configuration for Sequence Interactions
timing_patterns:
  continuous_processing:
    frame_interval: 33ms        # 30 FPS
    detection_interval: 50ms    # 20 FPS  
    queue_update: 100ms         # 10 Hz
    
  decision_making:
    state_preparation: 5ms
    neural_network: 5ms
    action_selection: 1ms
    command_transmission: 2ms
    total_decision_time: 13ms
    
  training_cycles:
    experience_storage: 1ms
    batch_sampling: 10ms
    forward_pass: 5ms
    backward_pass: 15ms
    parameter_update: 5ms
    total_training_step: 36ms
    
  error_recovery:
    error_detection: 100ms
    fallback_activation: 50ms
    recovery_attempt: 1000ms
    system_restart: 5000ms
```

### Component Interaction Patterns

```python
class SequenceOrchestrator:
    """Orchestrates component interactions for signal timing decisions."""
    
    def __init__(self):
        self.timing_constraints = {
            'frame_processing': 100,  # ms
            'decision_making': 50,    # ms
            'signal_execution': 20,   # ms
        }
        self.component_states = {}
        
    async def execute_decision_sequence(self):
        """Execute the main decision sequence with timing constraints."""
        start_time = time.time()
        
        # Step 1: Gather current state
        queue_state = await self.get_queue_state()
        
        # Step 2: Agent decision making
        action = await self.agent_decision(queue_state)
        
        # Step 3: Execute signal command
        await self.execute_signal_command(action)
        
        # Step 4: Log performance
        total_time = (time.time() - start_time) * 1000
        await self.log_sequence_performance(total_time)
        
    async def monitor_sequence_timing(self):
        """Monitor and ensure sequence timing requirements."""
        while True:
            performance = await self.collect_timing_metrics()
            if performance.decision_latency > self.timing_constraints['decision_making']:
                await self.trigger_performance_alert()
            await asyncio.sleep(1.0)
```

This comprehensive sequence diagram documentation provides detailed temporal views of how the adaptive traffic signal control system coordinates its components to make intelligent timing decisions while maintaining real-time performance requirements and handling various operational scenarios.