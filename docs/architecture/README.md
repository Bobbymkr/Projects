# 🏗️ System Architecture Documentation

This document provides a comprehensive overview of the Adaptive Traffic Signal Control System architecture, including detailed diagrams, component interactions, and data flow patterns.

## 📋 **Table of Contents**

- [High-Level System Architecture](#high-level-system-architecture)
- [Layered Architecture](#layered-architecture)
- [Component Interaction Diagrams](#component-interaction-diagrams)
- [Data Flow Architecture](#data-flow-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Multi-Agent Architecture](#multi-agent-architecture)

---

## 🎯 **High-Level System Architecture**

The Adaptive Traffic Signal Control System follows a **modular, layered architecture** that separates concerns across perception, decision-making, control, and monitoring layers.

```mermaid
graph TB
    subgraph "🌐 External Systems"
        VS[Video Sources]
        TS[Traffic Sensors]
        CS[City Systems]
    end
    
    subgraph "🎯 Perception Layer"
        VP[Video Pipeline]
        YD[YOLO Detection]
        QE[Queue Estimation]
        SE[Sensor Integration]
    end
    
    subgraph "🌍 Environment Layer"
        TE[Traffic Environment]
        SE2[SUMO Environment]
        VE[Video Environment]
        ME[MARL Environment]
    end
    
    subgraph "🧠 Decision Layer"
        DQN[DQN Agent]
        FC[Fuzzy Controller]
        WC[Webster Controller]
        GA[Genetic Algorithm]
        PSO[PSO Optimizer]
    end
    
    subgraph "🔮 Prediction Layer"
        TF[Traffic Forecaster]
        GNN[GNN Predictor]
        LSTM[LSTM Model]
    end
    
    subgraph "⚙️ Control Layer"
        SC[Signal Controller]
        TL[Traffic Lights]
        API[Control API]
    end
    
    subgraph "📊 Monitoring Layer"
        TB[TensorBoard]
        LOG[Logging System]
        METRICS[Metrics Collector]
        HEALTH[Health Monitor]
    end
    
    VS --> VP
    TS --> SE
    VP --> YD
    YD --> QE
    QE --> TE
    SE --> TE
    
    TE --> DQN
    SE2 --> DQN
    VE --> DQN
    ME --> DQN
    
    TF --> DQN
    GNN --> DQN
    LSTM --> TF
    
    DQN --> SC
    FC --> SC
    WC --> SC
    GA --> SC
    PSO --> SC
    
    SC --> TL
    SC --> API
    API --> CS
    
    DQN --> TB
    DQN --> LOG
    DQN --> METRICS
    SC --> HEALTH
    
    style DQN fill:#ff6b6b
    style TF fill:#4ecdc4
    style VP fill:#45b7d1
    style SC fill:#96ceb4
```

---

## 🏛️ **Layered Architecture**

### **Layer 1: Perception Layer** 🎯
**Purpose**: Convert raw sensor data into structured traffic information

```mermaid
graph LR
    subgraph "Input Sources"
        CAM[📹 Cameras]
        SENS[📡 Sensors]
        SIM[🎮 Simulation]
    end
    
    subgraph "Processing Pipeline"
        PREP[Preprocessing]
        ROI[ROI Extraction]
        DETECT[Object Detection]
        COUNT[Vehicle Counting]
    end
    
    subgraph "Output"
        QUEUES[Queue Lengths]
        FLOWS[Traffic Flows]
        STATES[Environment States]
    end
    
    CAM --> PREP
    SENS --> PREP
    SIM --> PREP
    
    PREP --> ROI
    ROI --> DETECT
    DETECT --> COUNT
    
    COUNT --> QUEUES
    COUNT --> FLOWS
    FLOWS --> STATES
```

### **Layer 2: Environment Layer** 🌍
**Purpose**: Provide standardized interfaces for different traffic scenarios

```mermaid
graph TB
    subgraph "Environment Abstractions"
        BASE[Gymnasium Base Environment]
        
        subgraph "Concrete Environments"
            TE[Traffic Environment<br/>• 4-lane intersection<br/>• Poisson arrivals<br/>• Queue dynamics]
            SE[SUMO Environment<br/>• Realistic simulation<br/>• TraCI interface<br/>• Complex networks]
            VE[Video Environment<br/>• Real-time video<br/>• Camera integration<br/>• Live processing]
            ME[MARL Environment<br/>• Multi-agent<br/>• Coordination<br/>• Communication]
        end
    end
    
    BASE --> TE
    BASE --> SE
    BASE --> VE
    BASE --> ME
    
    style BASE fill:#e74c3c
    style TE fill:#3498db
    style SE fill:#2ecc71
    style VE fill:#f39c12
    style ME fill:#9b59b6
```

### **Layer 3: Decision Layer** 🧠
**Purpose**: Intelligent decision making for optimal traffic control

```mermaid
graph TB
    subgraph "AI/ML Controllers"
        DQN[DQN Agent<br/>• Deep Q-Network<br/>• Experience Replay<br/>• Target Network]
        MARL[Multi-Agent RL<br/>• Distributed Control<br/>• Communication<br/>• Coordination]
    end
    
    subgraph "Classical Controllers"
        FUZZY[Fuzzy Logic<br/>• Linguistic Rules<br/>• Membership Functions<br/>• Interpretable]
        WEBSTER[Webster Method<br/>• Optimal Cycle<br/>• Capacity-based<br/>• Traditional]
    end
    
    subgraph "Optimization Controllers"
        GA[Genetic Algorithm<br/>• Population-based<br/>• Evolution Strategy<br/>• Global Search]
        PSO[Particle Swarm<br/>• Swarm Intelligence<br/>• Local Search<br/>• Convergence]
    end
    
    ENV_STATE[Environment State] --> DQN
    ENV_STATE --> MARL
    ENV_STATE --> FUZZY
    ENV_STATE --> WEBSTER
    ENV_STATE --> GA
    ENV_STATE --> PSO
    
    style DQN fill:#e74c3c
    style MARL fill:#8e44ad
    style FUZZY fill:#27ae60
    style WEBSTER fill:#f39c12
```

### **Layer 4: Control Layer** ⚙️
**Purpose**: Execute control decisions and interface with physical systems

```mermaid
graph LR
    subgraph "Control Interface"
        DECISIONS[Control Decisions]
        VALIDATOR[Action Validator]
        EXECUTOR[Control Executor]
    end
    
    subgraph "Output Systems"
        LIGHTS[Traffic Lights]
        API[External APIs]
        SIM[Simulation Control]
        LOG[Control Logging]
    end
    
    DECISIONS --> VALIDATOR
    VALIDATOR --> EXECUTOR
    
    EXECUTOR --> LIGHTS
    EXECUTOR --> API
    EXECUTOR --> SIM
    EXECUTOR --> LOG
```

---

## 🔄 **Component Interaction Diagrams**

### **Training Pipeline Interaction**

```mermaid
sequenceDiagram
    participant User
    participant Trainer
    participant Environment
    participant Agent
    participant ReplayBuffer
    participant QNetwork
    participant Metrics
    
    User->>Trainer: Start Training
    Trainer->>Environment: Reset
    Environment-->>Trainer: Initial State
    
    loop Training Episodes
        Trainer->>Agent: Get Action
        Agent->>QNetwork: Forward Pass
        QNetwork-->>Agent: Q-Values
        Agent-->>Trainer: Action
        
        Trainer->>Environment: Step(action)
        Environment-->>Trainer: Next State, Reward
        
        Trainer->>ReplayBuffer: Store Experience
        Trainer->>Agent: Learn
        
        Agent->>ReplayBuffer: Sample Batch
        ReplayBuffer-->>Agent: Experience Batch
        Agent->>QNetwork: Compute Loss
        QNetwork-->>Agent: Updated Weights
        
        Trainer->>Metrics: Record Episode
    end
    
    Trainer-->>User: Training Complete
```

### **Real-Time Inference Flow**

```mermaid
sequenceDiagram
    participant Camera
    participant VideoProcessor
    participant QueueEstimator
    participant Agent
    participant Controller
    participant TrafficLight
    
    Camera->>VideoProcessor: Video Frame
    VideoProcessor->>QueueEstimator: Processed Frame
    QueueEstimator->>QueueEstimator: YOLO Detection
    QueueEstimator-->>Agent: Queue Lengths
    
    Agent->>Agent: State Processing
    Agent->>Agent: Action Selection
    Agent-->>Controller: Control Decision
    
    Controller->>Controller: Validate Action
    Controller->>TrafficLight: Signal Command
    TrafficLight-->>Controller: Status Confirmation
    
    Controller-->>Agent: Execution Status
```

---

## 📊 **Data Flow Architecture**

### **Multi-Modal Data Integration**

```mermaid
graph TB
    subgraph "Data Sources"
        VIDEO[📹 Video Streams]
        SENSORS[📡 Traffic Sensors]
        WEATHER[🌤️ Weather Data]
        EVENTS[📅 Event Data]
    end
    
    subgraph "Data Processing"
        VP[Video Processing]
        SP[Sensor Processing]
        WP[Weather Processing]
        EP[Event Processing]
    end
    
    subgraph "Feature Extraction"
        QE[Queue Estimation]
        FE[Flow Estimation]
        DE[Density Estimation]
        WE[Weather Features]
    end
    
    subgraph "State Representation"
        FUSION[Data Fusion]
        NORM[Normalization]
        STATE[State Vector]
    end
    
    subgraph "Decision Making"
        AGENT[AI Agent]
        FORECAST[Forecasting]
        CONTROL[Control Logic]
    end
    
    VIDEO --> VP
    SENSORS --> SP
    WEATHER --> WP
    EVENTS --> EP
    
    VP --> QE
    SP --> FE
    SP --> DE
    WP --> WE
    
    QE --> FUSION
    FE --> FUSION
    DE --> FUSION
    WE --> FUSION
    
    FUSION --> NORM
    NORM --> STATE
    
    STATE --> AGENT
    STATE --> FORECAST
    FORECAST --> AGENT
    AGENT --> CONTROL
```

### **Information Flow in Multi-Agent System**

```mermaid
graph TB
    subgraph "Intersection A"
        A_SENS[Sensors A]
        A_AGENT[Agent A]
        A_CTRL[Controller A]
    end
    
    subgraph "Intersection B"
        B_SENS[Sensors B]
        B_AGENT[Agent B]
        B_CTRL[Controller B]
    end
    
    subgraph "Intersection C"
        C_SENS[Sensors C]
        C_AGENT[Agent C]
        C_CTRL[Controller C]
    end
    
    subgraph "Central Coordination"
        COORD[Coordination Layer]
        FORECAST[Global Forecasting]
        OPTIMIZER[Network Optimizer]
    end
    
    A_SENS --> A_AGENT
    B_SENS --> B_AGENT
    C_SENS --> C_AGENT
    
    A_AGENT <--> COORD
    B_AGENT <--> COORD
    C_AGENT <--> COORD
    
    COORD --> FORECAST
    FORECAST --> OPTIMIZER
    OPTIMIZER --> COORD
    
    A_AGENT --> A_CTRL
    B_AGENT --> B_CTRL
    C_AGENT --> C_CTRL
    
    A_AGENT -.-> B_AGENT
    B_AGENT -.-> C_AGENT
    C_AGENT -.-> A_AGENT
```

---

## 🚀 **Deployment Architecture**

### **Edge Deployment**

```mermaid
graph TB
    subgraph "Edge Device"
        CAMERA[📹 Camera Input]
        PROC[Edge Processor]
        LOCAL_AI[Local AI Model]
        CACHE[Local Cache]
        COMM[Communication Module]
    end
    
    subgraph "Local Network"
        GATEWAY[Network Gateway]
        LOCAL_DB[Local Database]
        BACKUP[Backup Controller]
    end
    
    subgraph "Cloud Infrastructure"
        CLOUD_AI[Cloud AI Models]
        GLOBAL_DB[Global Database]
        ANALYTICS[Analytics Engine]
        MONITOR[Monitoring]
    end
    
    CAMERA --> PROC
    PROC --> LOCAL_AI
    LOCAL_AI --> CACHE
    CACHE --> COMM
    
    COMM --> GATEWAY
    GATEWAY --> LOCAL_DB
    GATEWAY --> BACKUP
    
    GATEWAY <--> CLOUD_AI
    LOCAL_DB <--> GLOBAL_DB
    ANALYTICS --> MONITOR
```

### **Cloud-Native Architecture**

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Load Balancer]
    end
    
    subgraph "API Gateway"
        GATEWAY[API Gateway]
        AUTH[Authentication]
        RATE[Rate Limiting]
    end
    
    subgraph "Microservices"
        TRAFFIC[Traffic Service]
        AI[AI Service]
        VIDEO[Video Service]
        FORECAST[Forecast Service]
        CONTROL[Control Service]
    end
    
    subgraph "Data Layer"
        CACHE[Redis Cache]
        DB[PostgreSQL]
        STREAM[Kafka Streams]
        STORE[Object Storage]
    end
    
    subgraph "Infrastructure"
        MONITOR[Monitoring]
        LOG[Logging]
        TRACE[Tracing]
        ALERT[Alerting]
    end
    
    LB --> GATEWAY
    GATEWAY --> AUTH
    GATEWAY --> RATE
    
    GATEWAY --> TRAFFIC
    GATEWAY --> AI
    GATEWAY --> VIDEO
    GATEWAY --> FORECAST
    GATEWAY --> CONTROL
    
    TRAFFIC --> DB
    AI --> CACHE
    VIDEO --> STORE
    FORECAST --> STREAM
    CONTROL --> DB
    
    TRAFFIC --> MONITOR
    AI --> LOG
    VIDEO --> TRACE
    FORECAST --> ALERT
```

---

## 🤝 **Multi-Agent Architecture**

### **Distributed Agent Coordination**

```mermaid
graph TB
    subgraph "Agent Network"
        A1[Agent 1<br/>Intersection A]
        A2[Agent 2<br/>Intersection B]
        A3[Agent 3<br/>Intersection C]
        A4[Agent 4<br/>Intersection D]
    end
    
    subgraph "Communication Layer"
        MSG[Message Broker]
        SYNC[Synchronization]
        COORD[Coordination Protocol]
    end
    
    subgraph "Shared Resources"
        SHARED_MEM[Shared Memory]
        GLOBAL_STATE[Global State]
        POLICIES[Shared Policies]
    end
    
    subgraph "Coordination Strategies"
        CONSENSUS[Consensus Algorithm]
        AUCTION[Auction Mechanism]
        HIERARCHY[Hierarchical Control]
    end
    
    A1 <--> MSG
    A2 <--> MSG
    A3 <--> MSG
    A4 <--> MSG
    
    MSG --> SYNC
    SYNC --> COORD
    
    COORD <--> SHARED_MEM
    COORD <--> GLOBAL_STATE
    COORD <--> POLICIES
    
    COORD --> CONSENSUS
    COORD --> AUCTION
    COORD --> HIERARCHY
    
    A1 -.-> A2
    A2 -.-> A3
    A3 -.-> A4
    A4 -.-> A1
```

### **Hierarchical Multi-Agent Structure**

```mermaid
graph TB
    subgraph "Global Level"
        GLOBAL[Global Coordinator]
        STRATEGY[Global Strategy]
        PLANNING[Network Planning]
    end
    
    subgraph "Regional Level"
        R1[Regional Agent 1]
        R2[Regional Agent 2]
        R3[Regional Agent 3]
    end
    
    subgraph "Local Level"
        L1[Local Agent 1]
        L2[Local Agent 2]
        L3[Local Agent 3]
        L4[Local Agent 4]
        L5[Local Agent 5]
        L6[Local Agent 6]
    end
    
    subgraph "Intersection Level"
        I1[Intersection 1]
        I2[Intersection 2]
        I3[Intersection 3]
        I4[Intersection 4]
        I5[Intersection 5]
        I6[Intersection 6]
    end
    
    GLOBAL --> STRATEGY
    GLOBAL --> PLANNING
    
    GLOBAL --> R1
    GLOBAL --> R2
    GLOBAL --> R3
    
    R1 --> L1
    R1 --> L2
    R2 --> L3
    R2 --> L4
    R3 --> L5
    R3 --> L6
    
    L1 --> I1
    L2 --> I2
    L3 --> I3
    L4 --> I4
    L5 --> I5
    L6 --> I6
```

---

## 🔧 **Technology Stack Architecture**

```mermaid
graph TB
    subgraph "Frontend Layer"
        DASH[Monitoring Dashboard]
        WEB[Web Interface]
        MOBILE[Mobile App]
    end
    
    subgraph "API Layer"
        REST[REST APIs]
        GRAPHQL[GraphQL]
        WEBSOCKET[WebSocket]
        GRPC[gRPC]
    end
    
    subgraph "Business Logic"
        TRAFFIC[Traffic Logic]
        AI_ENGINE[AI Engine]
        OPTIMIZER[Optimization Engine]
        SIMULATOR[Simulation Engine]
    end
    
    subgraph "ML/AI Stack"
        PYTORCH[PyTorch]
        TENSORFLOW[TensorFlow]
        SKLEARN[Scikit-learn]
        OPENCV[OpenCV]
    end
    
    subgraph "Data Layer"
        POSTGRESQL[PostgreSQL]
        REDIS[Redis]
        MONGODB[MongoDB]
        INFLUXDB[InfluxDB]
    end
    
    subgraph "Infrastructure"
        DOCKER[Docker]
        KUBERNETES[Kubernetes]
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
    end
    
    DASH --> REST
    WEB --> GRAPHQL
    MOBILE --> WEBSOCKET
    
    REST --> TRAFFIC
    GRAPHQL --> AI_ENGINE
    WEBSOCKET --> OPTIMIZER
    GRPC --> SIMULATOR
    
    AI_ENGINE --> PYTORCH
    AI_ENGINE --> TENSORFLOW
    OPTIMIZER --> SKLEARN
    SIMULATOR --> OPENCV
    
    TRAFFIC --> POSTGRESQL
    AI_ENGINE --> REDIS
    OPTIMIZER --> MONGODB
    SIMULATOR --> INFLUXDB
    
    POSTGRESQL --> DOCKER
    REDIS --> KUBERNETES
    MONGODB --> PROMETHEUS
    INFLUXDB --> GRAFANA
```

---

## 📈 **Performance Architecture**

### **Scalability Patterns**

```mermaid
graph LR
    subgraph "Horizontal Scaling"
        LB[Load Balancer]
        S1[Service Instance 1]
        S2[Service Instance 2]
        S3[Service Instance N]
    end
    
    subgraph "Vertical Scaling"
        CPU[CPU Scaling]
        MEM[Memory Scaling]
        GPU[GPU Scaling]
    end
    
    subgraph "Data Scaling"
        SHARD[Database Sharding]
        REPLICA[Read Replicas]
        CACHE[Distributed Cache]
    end
    
    LB --> S1
    LB --> S2
    LB --> S3
    
    S1 --> CPU
    S1 --> MEM
    S1 --> GPU
    
    S1 --> SHARD
    S2 --> REPLICA
    S3 --> CACHE
```

---

## 🔒 **Security Architecture**

```mermaid
graph TB
    subgraph "Perimeter Security"
        FIREWALL[Firewall]
        WAF[Web Application Firewall]
        DDoS[DDoS Protection]
    end
    
    subgraph "Authentication & Authorization"
        AUTH[Authentication Service]
        RBAC[Role-Based Access Control]
        JWT[JWT Tokens]
    end
    
    subgraph "Data Security"
        ENCRYPT[Encryption at Rest]
        TLS[TLS in Transit]
        KEY[Key Management]
    end
    
    subgraph "Monitoring & Audit"
        SIEM[Security Information and Event Management]
        AUDIT[Audit Logging]
        ALERT[Security Alerts]
    end
    
    FIREWALL --> AUTH
    WAF --> RBAC
    DDoS --> JWT
    
    AUTH --> ENCRYPT
    RBAC --> TLS
    JWT --> KEY
    
    ENCRYPT --> SIEM
    TLS --> AUDIT
    KEY --> ALERT
```

---

## 📚 **Related Documentation**

- **[API Reference](../api/README.md)**: Detailed API documentation
- **[User Manual](../user-guide/README.md)**: User guidance and tutorials
- **[Developer Guide](../developer-guide/README.md)**: Development setup and practices
- **[Deployment Guide](../deployment/README.md)**: Production deployment instructions

---

## 🎯 **Design Principles**

### **1. Modularity**
- Clear separation of concerns
- Pluggable components
- Standard interfaces

### **2. Scalability**
- Horizontal and vertical scaling
- Distributed architecture
- Load balancing

### **3. Reliability**
- Fault tolerance
- Graceful degradation
- Circuit breakers

### **4. Performance**
- Optimized algorithms
- Efficient data structures
- Resource management

### **5. Security**
- Defense in depth
- Principle of least privilege
- Secure by design

This architecture provides a robust, scalable, and maintainable foundation for intelligent traffic signal control systems.