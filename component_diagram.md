# Component Diagram - Adaptive Traffic Signal Control System

This document provides comprehensive Component Diagrams for the Adaptive Traffic Signal Control System, showing detailed subsystems, their internal components, interfaces, and interactions. The system is organized into six major subsystems, each containing specialized components with well-defined interfaces.

## System Overview - Component Architecture

```mermaid
graph TB
    subgraph "Computer Vision Subsystem"
        CVS[Computer Vision Subsystem]
    end
    
    subgraph "Reinforcement Learning Subsystem"
        RLS[RL Subsystem]
    end
    
    subgraph "Environment Subsystem"
        ENS[Environment Subsystem]
    end
    
    subgraph "Forecasting Subsystem"
        FCS[Forecasting Subsystem]
    end
    
    subgraph "Interface Subsystem"
        IFS[Interface Subsystem]
    end
    
    subgraph "Storage Subsystem"
        STS[Storage Subsystem]
    end
    
    CVS -->|State Vectors| RLS
    RLS -->|Actions| ENS
    ENS -->|Rewards & States| RLS
    FCS -->|Predictions| RLS
    IFS -->|Configuration| CVS
    IFS -->|Configuration| RLS
    IFS -->|Configuration| FCS
    RLS -->|Metrics| IFS
    CVS -->|Recordings| STS
    RLS -->|Models| STS
    IFS -->|Configs| STS
    STS -->|Historical Data| FCS
```

## Computer Vision Subsystem

```mermaid
graph TB
    subgraph "Computer Vision Subsystem"
        subgraph "Video Input Components"
            VC[VideoCapture<br/>Multi-source Input]
            FB[FrameBuffer<br/>Threaded Storage]
            VP[VideoPreprocessor<br/>Resize/Normalize]
        end
        
        subgraph "Detection & Tracking"
            YD[YOLODetector<br/>Vehicle Detection]
            VT[VehicleTracker<br/>Multi-object Tracking]
            CF[ConfidenceFilter<br/>Quality Control]
        end
        
        subgraph "ROI Management"
            RM[ROIManager<br/>Region Configuration]
            RH[ROIHelper<br/>Polygon Operations]
            LA[LaneAnalyzer<br/>Traffic Flow Analysis]
        end
        
        subgraph "Queue Estimation"
            QE[QueueEstimator<br/>Count Vehicles]
            SA[StateAggregator<br/>Multi-lane Assembly]
            TS[TemporalSmoother<br/>Noise Reduction]
        end
        
        subgraph "Performance Monitoring"
            PM[PerformanceMonitor<br/>FPS & Latency]
            VL[VisionLogger<br/>Detection Events]
        end
    end
    
    %% Internal connections
    VC --> FB
    FB --> VP
    VP --> YD
    YD --> CF
    CF --> VT
    VT --> QE
    RM --> RH
    RH --> LA
    LA --> QE
    QE --> SA
    SA --> TS
    
    %% Monitoring connections
    YD --> PM
    VT --> PM
    QE --> VL
    PM --> VL
    
    %% External interfaces
    TS -->|State Vector| EXT_RL[To RL Subsystem]
    VL -->|Video Data| EXT_STORAGE[To Storage]
    EXT_CONFIG[From Interface] -->|ROI Config| RM
```

## Reinforcement Learning Subsystem

```mermaid
graph TB
    subgraph "Reinforcement Learning & Control Subsystem"
        subgraph "Model-Based RL"
            WM[WorldModel<br/>Dynamics Learning]
            MPC[ModelPredictiveControl<br/>Planning]
            MBA[ModelBasedRLAgent<br/>Policy Wrapper]
        end

        subgraph "Hierarchical RL"
            HLP[HighLevelPolicy<br/>Phase Selection]
            LLP[LowLevelPolicy<br/>Timing Control]
            HRA[HierarchicalRLAgent<br/>Coordinator]
        end

        subgraph "Deep RL Agents"
            TR[TransformerAgent<br/>Sequence Modeling]
            DQN[DQNAgent<br/>Value-based]
            MAML[MAMLAgent<br/>Meta-learning]
            REPT[ReptileAgent<br/>Meta-learning]
            HYB[HybridILRLAgent<br/>Imitation+RL]
        end

        subgraph "Probabilistic/Explainable"
            BAY[BayesianAgent<br/>Uncertainty]
            CAU[CausalAgent<br/>Causal Reasoning]
            NS[NeuroSymbolicAgent<br/>Constraints]
        end

        subgraph "Classical Controllers"
            FZ[FuzzyController<br/>Rule-based]
            WB[WebsterMethod<br/>Analytical]
        end

        subgraph "Learning & Ops"
            RB[Replay/TransitionBuffer]
            ES[EpsilonScheduler]
            OPT[Optimizer]
            TN[TargetNetwork]
            TL[TrainingLoop]
            EV[Evaluator]
            MS[ModelSaver]
            RL[RewardLogger]
        end
    end

    %% Internal connections
    MBA --> MPC
    MPC --> MBA

    HRA --> HLP
    HLP --> LLP
    LLP --> HRA

    DQN --> TN
    DQN --> RB
    TR --> RB
    HYB --> RB

    RB --> OPT
    OPT --> DQN
    OPT --> TR

    ES --> DQN

    TL --> DQN
    TL --> TR
    TL --> MBA
    TL --> HRA
    TL --> EV
    EV --> MS

    %% External interfaces
    EXT_CV[From CV Subsystem] -->|States| TR
    EXT_CV -->|States| DQN
    EXT_CV -->|States| MBA
    EXT_CV -->|States| HRA

    MBA -->|Actions| EXT_ENV[To Environment]
    HRA -->|Actions| EXT_ENV
    TR -->|Actions| EXT_ENV
    DQN -->|Actions| EXT_ENV

    EXT_ENV -->|Rewards/Transitions| RB
    MS -->|Models| EXT_STORAGE[To Storage]
    EXT_FORECAST[From Forecasting] -->|Predictions| TR
    EXT_FORECAST -->|Predictions| MBA
```

## Environment Subsystem

```mermaid
graph TB
    subgraph "Environment Subsystem"
        subgraph "Base Environment"
            TE[TrafficEnv<br/>Base Class]
            GI[GymnasiumInterface<br/>Standard API]
            AS_ENV[ActionSpace<br/>Discrete Actions]
            OS[ObservationSpace<br/>State Definition]
        end
        
        subgraph "Simulation Environments"
            SE[SumoEnv<br/>SUMO Integration]
            TI[TraCIInterface<br/>Command Bridge]
            SS[SimulationState<br/>Traffic State]
            SC[SimulationControl<br/>Step Management]
        end
        
        subgraph "Real-world Environment"
            VE[VideoEnv<br/>Camera Input]
            VI[VideoInterface<br/>Pipeline Bridge]
            RS[RealState<br/>Live Traffic]
            HC[HardwareControl<br/>Signal Interface]
        end
        
        subgraph "Multi-Agent Environment"
            ME[MarlEnv<br/>Multi-intersection]
            AC[AgentCoordinator<br/>Sync Management]
            IN[IntersectionNetwork<br/>Topology]
            CC[CommunicationChannel<br/>Agent Messages]
        end
        
        subgraph "Shared Logic"
            QL[QueueLogic<br/>Common Calculations]
            RF[RewardFunction<br/>Performance Metrics]
            ST[StateTransition<br/>Environment Updates]
            VF[ValidationFramework<br/>Action Validation]
        end
    end
    
    %% Inheritance relationships
    TE --> SE
    TE --> VE
    TE --> ME
    
    %% Internal connections
    SE --> TI
    TI --> SS
    SS --> SC
    
    VE --> VI
    VI --> RS
    RS --> HC
    
    ME --> AC
    AC --> IN
    IN --> CC
    
    %% Shared component usage
    SE --> QL
    VE --> QL
    ME --> QL
    QL --> RF
    RF --> ST
    ST --> VF
    
    %% External interfaces
    EXT_RL[From RL Subsystem] -->|Actions| GI
    GI -->|States/Rewards| EXT_RL
    EXT_CV[From CV Subsystem] -->|Video States| VI
    HC -->|Signal Commands| EXT_HARDWARE[To Traffic Controllers]
```

## Forecasting Subsystem

```mermaid
graph TB
    subgraph "Forecasting Subsystem"
        subgraph "Data Processing"
            DP[DataPreprocessor<br/>Time Series Prep]
            FE[FeatureExtractor<br/>Pattern Detection]
            SN[SequenceNormalizer<br/>Scale Adjustment]
            WM[WindowManager<br/>Sliding Windows]
        end
        
        subgraph "Model Components"
            LSTM[LSTMModel<br/>Temporal Prediction]
            CNN[CNNModel<br/>Spatial Features]
            GNN[GNNModel<br/>Graph Networks]
            HM[HybridModel<br/>CNN-LSTM Fusion]
        end
        
        subgraph "Training Pipeline"
            FT[ForecastTrainer<br/>Model Training]
            LF[LossFunction<br/>Prediction Error]
            OPT_FC[Optimizer<br/>Model Updates]
            VS[ValidationSplit<br/>Test Data]
        end
        
        subgraph "Prediction Engine"
            PI[PredictionInterface<br/>Inference API]
            BM[BatchManager<br/>Batch Processing]
            RT[RealTimePredictor<br/>Live Forecasting]
            PC[PredictionCache<br/>Result Storage]
        end
        
        subgraph "Model Management"
            MM[ModelManager<br/>Version Control]
            MC[ModelComparator<br/>Performance Eval]
            AS_FC[AutoSelector<br/>Best Model Choice]
        end
    end
    
    %% Internal connections
    DP --> FE
    FE --> SN
    SN --> WM
    WM --> LSTM
    WM --> CNN
    WM --> GNN
    
    LSTM --> HM
    CNN --> HM
    GNN --> HM
    
    FT --> LF
    LF --> OPT_FC
    OPT_FC --> LSTM
    OPT_FC --> CNN
    OPT_FC --> GNN
    FT --> VS
    
    HM --> PI
    PI --> BM
    BM --> RT
    RT --> PC
    
    MM --> MC
    MC --> AS_FC
    AS_FC --> PI
    
    %% External interfaces
    EXT_STORAGE[From Storage] -->|Historical Data| DP
    RT -->|Predictions| EXT_RL[To RL Subsystem]
    MM -->|Models| EXT_STORAGE
    EXT_CONFIG[From Interface] -->|Parameters| FT
```

## Interface Subsystem

```mermaid
graph TB
    subgraph "Interface Subsystem"
        subgraph "Web Server"
            WS[WebServer<br/>FastAPI App]
            RT_WS[Router<br/>Endpoint Management]
            MW[Middleware<br/>CORS/Auth/Metrics]
            SH[StaticHandler<br/>Frontend Assets]
        end
        
        subgraph "Real-time Communication"
            WSH[WebSocketHandler<br/>Live Updates]
            SSE[ServerSentEvents<br/>Push Notifications]
            MB[MessageBroker<br/>Pub/Sub System]
            EC[EventChannel<br/>System Events]
        end
        
        subgraph "Configuration Management"
            CM[ConfigManager<br/>System Settings]
            CV[ConfigValidator<br/>Schema Validation]
            CR[ConfigRenderer<br/>UI Forms]
            CB[ConfigBackup<br/>Version Control]
        end
        
        subgraph "Performance Monitoring"
            PM_IF[PerformanceMonitor<br/>System Metrics]
            HD[HealthDashboard<br/>Status Display]
            AM[AlertManager<br/>Notification System]
            MR[MetricsReporter<br/>Analytics Export]
        end
        
        subgraph "Authentication & Security"
            AU[Authenticator<br/>User Management]
            AZ[Authorizer<br/>Role-based Access]
            SV[SessionValidator<br/>Token Management]
            AL[AuditLogger<br/>Security Events]
        end
        
        subgraph "API Gateway"
            AG[APIGateway<br/>External Integration]
            RL_IF[RateLimiter<br/>Request Control]
            VL_IF[RequestValidator<br/>Input Sanitization]
            RS[ResponseSerializer<br/>Output Formatting]
        end
    end
    
    %% Internal connections
    WS --> RT_WS
    RT_WS --> MW
    MW --> SH
    
    WSH --> SSE
    SSE --> MB
    MB --> EC
    
    CM --> CV
    CV --> CR
    CR --> CB
    
    PM_IF --> HD
    HD --> AM
    AM --> MR
    
    AU --> AZ
    AZ --> SV
    SV --> AL
    
    AG --> RL_IF
    RL_IF --> VL_IF
    VL_IF --> RS
    
    %% Cross-subsystem connections
    MW --> AU
    RT_WS --> CM
    WSH --> PM_IF
    AG --> WS
    
    %% External interfaces
    CM -->|Configuration| EXT_ALL[To All Subsystems]
    EXT_ALL -->|Metrics| PM_IF
    WSH -->|Live Data| EXT_CLIENT[To Web Dashboard]
    AG -->|API Access| EXT_EXTERNAL[External Systems]
```

## Storage Subsystem

```mermaid
graph TB
    subgraph "Storage Subsystem"
        subgraph "Model Storage"
            MS[ModelStore<br/>Neural Network Weights]
            CP[CheckpointManager<br/>Version Control]
            MC_ST[ModelCompression<br/>Size Optimization]
            ML[ModelLoader<br/>Runtime Loading]
        end
        
        subgraph "Performance Logging"
            PL[PerformanceLogger<br/>Metrics Storage]
            TS[TimeSeriesDB<br/>Historical Data]
            AG_ST[Aggregator<br/>Data Summarization]
            EX[Exporter<br/>Data Export]
        end
        
        subgraph "Configuration Repository"
            CR_ST[ConfigRepository<br/>Settings Storage]
            VS_ST[VersionStore<br/>Config History]
            BM_ST[BackupManager<br/>Disaster Recovery]
            SY[Synchronizer<br/>Multi-instance Sync]
        end
        
        subgraph "File Storage"
            FS[FileStorage<br/>Video/Image Files]
            CM_ST[CacheManager<br/>Temporary Storage]
            CD[CompressionDriver<br/>Space Optimization]
            RP[RetentionPolicy<br/>Cleanup Rules]
        end
        
        subgraph "Database Layer"
            DM[DatabaseManager<br/>Connection Pool]
            QE_ST[QueryEngine<br/>Data Retrieval]
            IM[IndexManager<br/>Performance Optimization]
            BK[BackupKernel<br/>Data Protection]
        end
        
        subgraph "Data Pipeline"
            DI[DataIngestion<br/>Batch Processing]
            ET[ETLTransformer<br/>Data Processing]
            VL_ST[ValidationLayer<br/>Data Quality]
            AM_ST[ArchiveManager<br/>Long-term Storage]
        end
    end
    
    %% Internal connections
    MS --> CP
    CP --> MC_ST
    MC_ST --> ML
    
    PL --> TS
    TS --> AG_ST
    AG_ST --> EX
    
    CR_ST --> VS_ST
    VS_ST --> BM_ST
    BM_ST --> SY
    
    FS --> CM_ST
    CM_ST --> CD
    CD --> RP
    
    DM --> QE_ST
    QE_ST --> IM
    IM --> BK
    
    DI --> ET
    ET --> VL_ST
    VL_ST --> AM_ST
    
    %% Cross-component connections
    MS --> DM
    PL --> DM
    CR_ST --> DM
    FS --> DI
    
    %% External interfaces
    EXT_RL[From RL Subsystem] -->|Models| MS
    EXT_CV[From CV Subsystem] -->|Videos| FS
    EXT_IF[From Interface] -->|Configs| CR_ST
    EXT_ALL[All Subsystems] -->|Metrics| PL
    ML -->|Models| EXT_RL
    QE_ST -->|Historical Data| EXT_FORECAST[To Forecasting]
```

## Component Interaction Patterns

### Request-Response Pattern

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant WS as Web Server
    participant CM as Config Manager
    participant DQN as DQN Agent
    participant Storage as Model Store

    Client->>WS: POST /api/config/update
    WS->>CM: validate_config(new_config)
    CM->>Storage: save_config(validated_config)
    Storage-->>CM: config_saved
    CM-->>WS: validation_result
    WS->>DQN: update_configuration(new_config)
    DQN-->>WS: config_applied
    WS-->>Client: 200 OK + status
```

### Event-Driven Pattern

```mermaid
sequenceDiagram
    participant CV as Computer Vision
    participant RL as RL Agent
    participant Env as Environment
    participant WS as WebSocket Handler
    participant Client as Dashboard

    CV->>RL: state_update(queue_lengths)
    RL->>Env: take_action(green_duration)
    Env->>RL: reward_feedback(performance)
    RL->>WS: emit_event("performance_update", metrics)
    WS->>Client: push_notification(real_time_data)
```

### Data Pipeline Pattern

```mermaid
graph LR
    subgraph "Input Stage"
        VS[Video Source]
        HD[Historical Data]
    end
    
    subgraph "Processing Stage"
        CV[Computer Vision]
        FC[Forecasting]
        RL[RL Processing]
    end
    
    subgraph "Output Stage"
        ENV[Environment Control]
        STORE[Storage]
        UI[User Interface]
    end
    
    VS --> CV
    HD --> FC
    CV --> RL
    FC --> RL
    RL --> ENV
    RL --> STORE
    RL --> UI
```

## Component Dependencies

### Dependency Graph

```mermaid
graph TD
    subgraph "Core Dependencies"
        CV[Computer Vision<br/>Independent]
        RL[Reinforcement Learning<br/>Depends: CV, Forecasting]
        ENV[Environment<br/>Depends: CV]
        FC[Forecasting<br/>Depends: Storage]
    end
    
    subgraph "Infrastructure Dependencies"
        IF[Interface<br/>Depends: All Core]
        ST[Storage<br/>Independent]
    end
    
    CV -->|State Data| RL
    FC -->|Predictions| RL
    RL -->|Actions| ENV
    ENV -->|Feedback| RL
    
    CV -->|Video Data| ST
    RL -->|Models/Metrics| ST
    FC -->|Models| ST
    IF -->|Configs| ST
    
    ST -->|Historical Data| FC
    ST -->|Saved Models| RL
    
    IF -->|Configuration| CV
    IF -->|Configuration| RL
    IF -->|Configuration| FC
    
    CV -->|Metrics| IF
    RL -->|Metrics| IF
    ENV -->|Status| IF
    FC -->|Results| IF
```

## Interface Specifications

### Computer Vision Interface

```yaml
ComputerVisionInterface:
  inputs:
    - video_stream: VideoStream
    - roi_config: ROIConfiguration
  outputs:
    - state_vector: float[4]
    - detection_metadata: DetectionResults
  methods:
    - process_frame(frame) -> StateVector
    - configure_roi(roi_config) -> bool
    - get_performance_metrics() -> Metrics
```

### Reinforcement Learning Interface

```yaml
ReinforcementLearningInterface:
  inputs:
    - state_vector: float[4]
    - forecast_data: ForecastResults
    - configuration: RLConfig
  outputs:
    - action: int
    - q_values: float[12]
    - training_metrics: TrainingResults
  methods:
    - select_action(state) -> int
    - train_step(batch) -> float
    - save_model(path) -> bool
    - load_model(path) -> bool
```

### Environment Interface

```yaml
EnvironmentInterface:
  inputs:
    - action: int
    - external_state: Any
  outputs:
    - observation: float[4]
    - reward: float
    - terminated: bool
    - info: Dict
  methods:
    - reset() -> Observation
    - step(action) -> (obs, reward, term, trunc, info)
    - render() -> Any
```

## Quality Attributes by Component

### Performance Characteristics

| **Component** | **Latency** | **Throughput** | **Memory** | **CPU** |
|---------------|-------------|----------------|------------|---------|
| VideoCapture | 33ms | 30 FPS | 100MB | 10% |
| YOLODetector | 50ms | 20 FPS | 2GB | 80% |
| DQNAgent | 5ms | 200 Hz | 50MB | 15% |
| WebServer | 10ms | 1000 req/s | 200MB | 5% |
| ModelStore | 100ms | 10 ops/s | 1GB | 5% |

### Reliability Patterns

```mermaid
graph TB
    subgraph "Fault Tolerance"
        FD[Failure Detection]
        FR[Failure Recovery]
        FB[Fallback Mechanisms]
    end
    
    subgraph "Health Monitoring"
        HC[Health Checks]
        HB[Heartbeat Monitoring]
        AM[Alert Management]
    end
    
    subgraph "Data Integrity"
        VL[Validation Layers]
        BC[Backup Copies]
        CS[Checksums]
    end
    
    FD --> FR
    FR --> FB
    HC --> HB
    HB --> AM
    VL --> BC
    BC --> CS
```

This comprehensive component diagram provides a detailed view of the system's internal structure, showing how each subsystem is organized and how components interact through well-defined interfaces. Each subsystem encapsulates related functionality while maintaining loose coupling with other subsystems through standardized interfaces.