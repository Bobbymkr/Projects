# Data Flow Diagram - Adaptive Traffic Signal Control System

This document provides comprehensive Data Flow Diagrams (DFDs) for the Adaptive Traffic Signal Control System, illustrating how information moves through the system from input sources to output destinations across different operational modes.

## Overview

The system processes multiple data streams including video feeds, simulation data, historical traffic patterns, and configuration parameters to produce optimal traffic signal control decisions. Data flows through various processing stages with real-time constraints and quality assurance mechanisms.

## Context-Level Data Flow Diagram (Level 0)

```mermaid
graph TB
    subgraph "External Entities"
        CAM[Camera Systems<br/>Video Sources]
        TC[Traffic Controllers<br/>Signal Hardware]
        OPR[System Operators<br/>Traffic Engineers]
        ENV[Environment<br/>Weather/Events]
    end
    
    subgraph "System Boundary"
        ATSCS[Adaptive Traffic<br/>Signal Control System]
    end
    
    CAM -->|Video Streams<br/>Real-time Footage| ATSCS
    ENV -->|Weather Data<br/>Event Information| ATSCS
    OPR -->|Configuration<br/>Parameters<br/>ROI Settings| ATSCS
    
    ATSCS -->|Signal Commands<br/>Timing Instructions| TC
    ATSCS -->|Performance Reports<br/>System Status<br/>Analytics| OPR
    
    TC -->|Status Feedback<br/>Current State| ATSCS
```

## Level 1 Data Flow Diagram - System Overview

```mermaid
graph TB
    subgraph "Input Sources"
        VS[Video Sources<br/>Cameras/Files/RTSP]
        HD[Historical Data<br/>Traffic Patterns]
        CF[Configuration Files<br/>ROI/Parameters]
        SD[Simulation Data<br/>SUMO Environment]
    end
    
    subgraph "Core Processing Pipeline"
        VP[1.0<br/>Video Processing<br/>Pipeline]
        SA[2.0<br/>State Assembly<br/>& Normalization]
        IA[3.0<br/>Intelligence Agent<br/>DQN Decision Making]
        SC[4.0<br/>Signal Control<br/>& Execution]
    end
    
    subgraph "Output Destinations"
        SH[Signal Hardware<br/>Traffic Controllers]
        DB[Dashboard<br/>Web Interface]
        PL[Performance Logs<br/>Analytics Database]
        MS[Model Storage<br/>Checkpoints]
    end
    
    subgraph "Data Stores"
        DS1[(D1: Configuration<br/>Repository)]
        DS2[(D2: Performance<br/>Metrics)]
        DS3[(D3: Model<br/>Checkpoints)]
        DS4[(D4: Video<br/>Recordings)]
    end
    
    %% Input flows
    VS -->|Raw Video Frames| VP
    HD -->|Historical Patterns| SA
    CF -->|System Configuration| DS1
    SD -->|Simulation State| SA
    
    %% Processing flows
    VP -->|Vehicle Detections<br/>Queue Estimates| SA
    SA -->|State Vector<br/>Normalized Data| IA
    IA -->|Action Selection<br/>Green Duration| SC
    
    %% Output flows
    SC -->|Control Commands| SH
    IA -->|Performance Metrics| DB
    IA -->|Training Data| PL
    IA -->|Model Weights| MS
    VP -->|Video Recordings| DS4
    
    %% Data store interactions
    DS1 -->|ROI Configurations<br/>Parameters| VP
    DS1 -->|Agent Configuration| IA
    DS2 -->|Historical Metrics| DB
    DS3 -->|Trained Models| IA
    
    IA -->|Experience Replay| DS2
    IA -->|Model Updates| DS3
```

## Level 2 Data Flow Diagram - Video Processing Pipeline

```mermaid
graph TB
    subgraph "Video Input Processing"
        VIS[Video Input<br/>Stream]
        FC[2.1<br/>Frame Capture<br/>& Buffering]
        PP[2.2<br/>Preprocessing<br/>Resize/Normalize]
        ROI[2.3<br/>ROI Application<br/>Lane Segmentation]
    end
    
    subgraph "Computer Vision Pipeline"
        VD[2.4<br/>Vehicle Detection<br/>YOLOv8]
        OT[2.5<br/>Object Tracking<br/>Multi-frame]
        QE[2.6<br/>Queue Estimation<br/>Stationary Analysis]
    end
    
    subgraph "State Construction"
        QA[2.7<br/>Queue Aggregation<br/>Per Lane]
        SN[2.8<br/>State Normalization<br/>Vector Assembly]
        TS[2.9<br/>Temporal Smoothing<br/>Noise Reduction]
    end
    
    subgraph "Data Stores"
        DS1[(D1: ROI<br/>Configurations)]
        DS4[(D4: Frame<br/>Buffer)]
        DS5[(D5: Detection<br/>Cache)]
    end
    
    VIS -->|Raw Frames| FC
    FC -->|Buffered Frames| PP
    PP -->|Processed Frames| ROI
    ROI -->|Lane Regions| VD
    
    VD -->|Vehicle Bboxes<br/>Confidence Scores| OT
    OT -->|Tracked Objects<br/>Trajectories| QE
    QE -->|Queue Counts<br/>Per Lane| QA
    
    QA -->|Aggregated Counts| SN
    SN -->|State Vector| TS
    TS -->|Smoothed State<br/>Traffic Conditions| SA[State Assembly]
    
    %% Data store interactions
    DS1 -->|ROI Polygons<br/>Lane Definitions| ROI
    FC -->|Frame Storage| DS4
    VD -->|Detection Results| DS5
    DS5 -->|Previous Detections| OT
```

## Level 2 Data Flow Diagram - Intelligence Layer

```mermaid
graph TB
    subgraph "State Processing"
        SI[State Input<br/>From Vision/Simulation]
        SF[3.1<br/>State Fusion<br/>Multi-source Integration]
        FP[3.2<br/>Feature Processing<br/>Normalization]
    end
    
    subgraph "Decision Making"
        NN[3.3<br/>Neural Network<br/>Q-Value Computation]
        AS[3.4<br/>Action Selection<br/>ε-greedy Strategy]
        AP[3.5<br/>Action Processing<br/>Duration Mapping]
    end
    
    subgraph "Learning System"
        ER[3.6<br/>Experience Replay<br/>Buffer Management]
        TN[3.7<br/>Target Network<br/>Stability Control]
        LO[3.8<br/>Loss Optimization<br/>Gradient Descent]
    end
    
    subgraph "Forecasting Integration"
        TF[3.9<br/>Traffic Forecasting<br/>LSTM Prediction]
        PA[3.10<br/>Predictive Analysis<br/>Future State Estimation]
    end
    
    subgraph "Data Stores"
        DS3[(D3: Model<br/>Weights)]
        DS6[(D6: Replay<br/>Buffer)]
        DS7[(D7: Training<br/>History)]
    end
    
    SI -->|Current State| SF
    SF -->|Fused State| FP
    FP -->|Processed Features| NN
    FP -->|Historical Sequence| TF
    
    NN -->|Q-Values| AS
    TF -->|Future Predictions| PA
    PA -->|Augmented State| NN
    
    AS -->|Selected Action| AP
    AP -->|Green Duration<br/>Signal Timing| SC[Signal Control]
    
    %% Learning flows
    SF -->|State-Action-Reward| ER
    ER -->|Training Batch| LO
    LO -->|Parameter Updates| NN
    NN -->|Weight Copy| TN
    
    %% Data store interactions
    DS3 -->|Trained Weights| NN
    DS3 -->|Target Weights| TN
    ER -->|Experiences| DS6
    DS6 -->|Replay Samples| ER
    LO -->|Training Metrics| DS7
    NN -->|Updated Weights| DS3
```

## Real-Time Processing Data Flow

```mermaid
sequenceDiagram
    participant Camera as Camera Feed
    participant Vision as Vision Pipeline
    participant Buffer as Frame Buffer
    participant YOLO as YOLO Detector
    participant Tracker as Object Tracker
    participant Agent as DQN Agent
    participant Controller as Signal Controller
    participant Logger as Performance Logger

    Camera->>Vision: Video Stream (30 FPS)
    Vision->>Buffer: Frame Capture
    Buffer->>Vision: Latest Frame
    Vision->>YOLO: Preprocessed Frame
    YOLO->>Tracker: Vehicle Detections
    Tracker->>Vision: Tracked Objects
    Vision->>Agent: Queue State Vector
    
    Agent->>Agent: Q-Value Computation
    Agent->>Agent: Action Selection
    Agent->>Controller: Green Duration Command
    Controller->>Agent: Status Confirmation
    
    Agent->>Logger: Performance Metrics
    Agent->>Buffer: Experience Replay
    
    Note over Camera, Logger: Processing Cycle: ~100ms
    Note over Agent, Controller: Decision Latency: <50ms
```

## Training Mode Data Flow

```mermaid
graph LR
    subgraph "Simulation Environment"
        SUMO[SUMO Simulation<br/>Traffic Microscopy]
        TI[TraCI Interface<br/>Command Bridge]
        ENV[Environment State<br/>Queue/Wait Times]
    end
    
    subgraph "Training Pipeline"
        AGENT[DQN Agent<br/>Learning]
        RB[Replay Buffer<br/>Experience Storage]
        TN[Target Network<br/>Stable Targets]
        OPT[Optimizer<br/>Adam/SGD]
    end
    
    subgraph "Monitoring"
        TB[TensorBoard<br/>Metrics Visualization]
        LOG[Training Logs<br/>Performance Tracking]
        SAVE[Model Checkpoints<br/>Weight Persistence]
    end
    
    SUMO -->|Traffic State| TI
    TI -->|Observations| ENV
    ENV -->|State Vector| AGENT
    AGENT -->|Actions| ENV
    ENV -->|Rewards| AGENT
    
    AGENT -->|Experiences| RB
    RB -->|Training Batch| AGENT
    AGENT -->|Q-Values| OPT
    TN -->|Target Q-Values| OPT
    OPT -->|Weight Updates| AGENT
    
    AGENT -->|Training Metrics| TB
    AGENT -->|Performance Data| LOG
    AGENT -->|Model Weights| SAVE
```

## Data Transformation Processes

### Video Frame Processing Pipeline

```mermaid
flowchart TD
    A[Raw Video Frame<br/>1920x1080 BGR] --> B[Frame Resize<br/>640x480]
    B --> C[ROI Extraction<br/>Lane Polygons]
    C --> D[YOLOv8 Detection<br/>Vehicle Bounding Boxes]
    D --> E[Confidence Filtering<br/>Threshold: 0.5]
    E --> F[Object Tracking<br/>Centroid Association]
    F --> G[Motion Analysis<br/>Velocity Estimation]
    G --> H[Queue Classification<br/>Stationary Vehicles]
    H --> I[Lane Aggregation<br/>Count per ROI]
    I --> J[State Vector<br/>[q1, q2, q3, q4]]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#9f9,stroke:#333,stroke-width:2px
```

### State Vector Construction

```mermaid
flowchart LR
    subgraph "Input Sources"
        VQ[Vision Queues<br/>Real-time Detection]
        SQ[Simulation Queues<br/>SUMO Environment]
        FQ[Forecast Queues<br/>LSTM Prediction]
        HQ[Historical Queues<br/>Pattern Analysis]
    end
    
    subgraph "Processing"
        AGG[Aggregation<br/>Multi-source Fusion]
        NORM[Normalization<br/>0-1 Scaling]
        SMOOTH[Smoothing<br/>Temporal Filter]
    end
    
    subgraph "Output"
        SV[State Vector<br/>4D Float Array]
    end
    
    VQ --> AGG
    SQ --> AGG
    FQ --> AGG
    HQ --> AGG
    
    AGG --> NORM
    NORM --> SMOOTH
    SMOOTH --> SV
```

## Data Storage and Persistence

```mermaid
graph TB
    subgraph "Operational Data"
        RT[Real-time State<br/>Current Conditions]
        ACT[Action History<br/>Decision Log]
        REW[Reward Signals<br/>Performance Metrics]
    end
    
    subgraph "Training Data"
        EXP[Experience Replay<br/>State-Action-Reward]
        MODEL[Model Checkpoints<br/>Neural Network Weights]
        META[Training Metadata<br/>Hyperparameters]
    end
    
    subgraph "Configuration Data"
        ROI_CFG[ROI Configurations<br/>Lane Definitions]
        AGENT_CFG[Agent Parameters<br/>Learning Settings]
        SYS_CFG[System Settings<br/>Hardware Interface]
    end
    
    subgraph "Storage Systems"
        TS_DB[(Time Series DB<br/>InfluxDB)]
        FILE_SYS[(File System<br/>Local/Cloud)]
        CONFIG_DB[(Config Store<br/>JSON/YAML)]
    end
    
    RT --> TS_DB
    ACT --> TS_DB
    REW --> TS_DB
    
    EXP --> FILE_SYS
    MODEL --> FILE_SYS
    META --> FILE_SYS
    
    ROI_CFG --> CONFIG_DB
    AGENT_CFG --> CONFIG_DB
    SYS_CFG --> CONFIG_DB
```

## Performance and Quality Metrics

### Data Quality Assurance

```mermaid
flowchart TD
    INPUT[Input Data<br/>Raw Sensors] --> VALID[Validation<br/>Range/Type Checks]
    VALID --> FILTER[Filtering<br/>Outlier Removal]
    FILTER --> SMOOTH[Smoothing<br/>Temporal Consistency]
    SMOOTH --> QUALITY[Quality Assessment<br/>Confidence Scoring]
    QUALITY --> OUTPUT[Output Data<br/>Validated Stream]
    
    QUALITY -->|Low Quality| ALERT[Quality Alert<br/>System Notification]
    ALERT --> FALLBACK[Fallback Mode<br/>Default Behavior]
```

### Processing Pipeline Metrics

| **Stage** | **Input** | **Output** | **Latency** | **Throughput** |
|-----------|-----------|------------|-------------|----------------|
| Frame Capture | Video Stream | Raw Frames | 33ms (30 FPS) | 30 frames/sec |
| Preprocessing | Raw Frames | Processed Frames | 5ms | 200 frames/sec |
| YOLO Detection | Processed Frames | Bounding Boxes | 50ms | 20 frames/sec |
| Object Tracking | Detections | Tracks | 10ms | 100 tracks/sec |
| Queue Estimation | Tracks | Queue Counts | 2ms | 500 counts/sec |
| State Assembly | Queue Counts | State Vector | 1ms | 1000 vectors/sec |
| DQN Inference | State Vector | Action | 5ms | 200 actions/sec |
| Signal Control | Action | Commands | 10ms | 100 commands/sec |

## Error Handling and Data Recovery

```mermaid
graph TB
    subgraph "Error Detection"
        MD[Missing Data<br/>Sensor Failure]
        CD[Corrupted Data<br/>Transmission Error]
        OD[Outdated Data<br/>Timestamp Issues]
    end
    
    subgraph "Recovery Mechanisms"
        INTERP[Interpolation<br/>Estimate Missing Values]
        RETRY[Retry Logic<br/>Re-request Data]
        FALLBACK[Fallback Data<br/>Default/Historical]
        CACHE[Cached Data<br/>Previous Valid State]
    end
    
    subgraph "Quality Control"
        QC[Quality Check<br/>Validation Rules]
        ALERT[Alert System<br/>Notification]
        LOG[Error Logging<br/>Audit Trail]
    end
    
    MD --> INTERP
    MD --> FALLBACK
    CD --> RETRY
    CD --> CACHE
    OD --> CACHE
    
    INTERP --> QC
    RETRY --> QC
    FALLBACK --> QC
    CACHE --> QC
    
    QC --> ALERT
    QC --> LOG
```

## Real-Time Constraints and Optimization

### Timing Requirements

```mermaid
gantt
    title Processing Timeline (100ms cycle)
    dateFormat X
    axisFormat %L
    
    section Video Pipeline
    Frame Capture     :a1, 0, 10
    Preprocessing     :a2, after a1, 15
    YOLO Detection    :a3, after a2, 50
    Object Tracking   :a4, after a3, 10
    
    section Decision Making
    State Assembly    :b1, after a4, 5
    DQN Inference     :b2, after b1, 15
    Action Selection  :b3, after b2, 3
    
    section Signal Control
    Command Generation :c1, after b3, 5
    Hardware Interface :c2, after c1, 7
```

### Optimization Strategies

1. **Parallel Processing**: Multi-threaded video processing pipeline
2. **Memory Management**: Efficient buffer allocation and reuse
3. **Model Optimization**: Quantized neural networks for faster inference
4. **Caching**: Frequently accessed configuration and model data
5. **Batch Processing**: Group operations for GPU acceleration

This comprehensive data flow documentation provides a complete picture of how information moves through the adaptive traffic control system, enabling efficient maintenance, debugging, and system optimization.