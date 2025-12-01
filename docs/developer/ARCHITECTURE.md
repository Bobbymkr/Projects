# Architecture Overview
## Adaptive Traffic Signal Control System

Comprehensive architecture documentation for developers and system architects.

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Dashboard  │  │  Mobile App  │  │  External    │     │
│  │   (React)    │  │              │  │  Systems     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │         FastAPI Application                        │    │
│  │  • REST API (v1)                                   │    │
│  │  • GraphQL API                                     │    │
│  │  • WebSocket Support                               │    │
│  │  • Authentication & Authorization                  │    │
│  │  • Rate Limiting                                   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Traffic     │  │  Analytics   │  │  Research    │     │
│  │  Controller  │  │  Service     │  │  Platform    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     AI/ML Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  DQN Agent   │  │  GNN Models  │  │  Forecasting │     │
│  │  MARL        │  │  Transformers│  │  Models      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Infrastructure                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL  │  │    Redis     │  │  SUMO        │     │
│  │  (Primary)   │  │   (Cache)    │  │  Simulation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. API Layer (`src/api/`)

**Purpose**: Entry point for all external requests

**Components**:
- **main.py**: FastAPI application setup
- **routes/**: API endpoint definitions
  - `traffic.py`: Traffic control endpoints
  - `metrics.py`: Performance metrics
  - `analytics.py`: Analytics and reporting
  - `system.py`: System health and status
  - `auth.py`: Authentication endpoints
  - `graphql.py`: GraphQL API
- **services/**: Business logic services
- **schemas.py**: Pydantic models for validation
- **middleware/**: Request/response middleware
- **dependencies.py**: FastAPI dependencies

**Key Features**:
- OpenAPI 3.0 documentation
- Automatic request validation
- Rate limiting
- Authentication & authorization
- WebSocket support
- Error handling

---

### 2. Traffic Control Layer (`src/control/`)

**Purpose**: Core traffic signal control logic

**Components**:
- **fuzzy_control.py**: Fuzzy logic controller
- **webster_method.py**: Traditional Webster's method
- **adaptive_controller.py**: Adaptive control strategies

**Design Patterns**:
- Strategy pattern for algorithm selection
- Factory pattern for controller creation
- Observer pattern for state changes

---

### 3. Reinforcement Learning Layer (`src/rl/`)

**Purpose**: Deep RL agents for intelligent decision making

**Components**:
- **agents/**: RL agent implementations
  - `dqn_agent.py`: Deep Q-Network agent
  - `marl_agent.py`: Multi-agent RL
  - `hierarchical_agent.py`: Hierarchical RL
- **environments/**: Simulation environments
  - `traffic_env.py`: Single intersection environment
  - `marl_env.py`: Multi-agent environment
- **networks/**: Neural network architectures
  - `dqn_network.py`: DQN network
  - `gnn_network.py`: Graph neural network
- **training/**: Training scripts and utilities

**Key Features**:
- Multiple RL algorithms
- Custom reward functions
- Experience replay
- Target networks
- Multi-agent coordination

---

### 4. Forecasting Layer (`src/forecasting/`)

**Purpose**: Traffic prediction and forecasting

**Components**:
- **models/**: Forecasting models
  - `cnn_lstm.py`: CNN-LSTM hybrid model
  - `transformer_model.py`: Transformer-based model
  - `gnn_forecast.py`: GNN-based forecasting
- **data/**: Data preprocessing and preparation
- **evaluation/**: Model evaluation metrics

---

### 5. Research Platform (`src/research/`)

**Purpose**: Experimental algorithms and research tools

**Components**:
- **algorithms/**: Novel algorithm implementations
  - `imitation_learning.py`: Imitation learning
  - `model_based_rl.py`: Model-based RL
  - `federated_learning.py`: Federated learning coordinator
- **mlflow_integration.py**: Experiment tracking
- **optuna_integration.py**: Hyperparameter optimization
- **explainability/**: XAI tools (SHAP, LIME, Counterfactual)

---

### 6. Dashboard (`dashboard/`)

**Purpose**: Web-based user interface

**Technology Stack**:
- React 18 with TypeScript
- Ant Design UI components
- Redux for state management
- WebSocket for real-time updates
- ECharts for data visualization

**Components**:
- Executive dashboard
- Operations dashboard
- Analytics dashboard
- System monitoring

---

## Data Flow

### Request Flow

1. **Client Request** → API Gateway
2. **Authentication** → Validate token/credentials
3. **Rate Limiting** → Check request limits
4. **Request Validation** → Validate request schema
5. **Service Layer** → Execute business logic
6. **AI/ML Layer** → Process with ML models (if needed)
7. **Response** → Format and return response

### Real-Time Updates Flow

1. **State Change** → Traffic state changes
2. **Event Publisher** → Publish event to queue
3. **WebSocket Handler** → Broadcast to connected clients
4. **Client Update** → Dashboard receives update

---

## Design Patterns

### 1. Service Layer Pattern

Separates business logic from API endpoints:

```python
# API Route
@router.post("/traffic/decision")
async def make_decision(request: TrafficDecisionRequest):
    return await traffic_service.make_decision(request)

# Service Layer
class TrafficService:
    async def make_decision(self, request):
        # Business logic here
        pass
```

### 2. Repository Pattern (Future)

Abstraction layer for data access:

```python
class IntersectionRepository:
    async def get_by_id(self, id: str) -> Intersection:
        pass
```

### 3. Factory Pattern

Creates controllers based on algorithm type:

```python
controller = ControllerFactory.create(algorithm_type="dqn")
```

### 4. Observer Pattern

Notifies subscribers of state changes:

```python
event_publisher.subscribe(WebSocketHandler)
```

---

## Technology Stack

### Backend
- **Python 3.10+**: Core language
- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **SQLAlchemy**: ORM (if using database)
- **Redis**: Caching and queues
- **PyTorch**: Deep learning framework

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Ant Design**: UI components
- **Redux**: State management
- **WebSocket**: Real-time updates

### ML/AI
- **PyTorch**: Neural networks
- **Gym**: RL environments
- **MLflow**: Experiment tracking
- **Optuna**: Hyperparameter optimization
- **SHAP/LIME**: Explainability

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **PostgreSQL**: Primary database
- **Redis**: Cache and queues
- **Prometheus**: Metrics
- **Grafana**: Visualization

---

## Scalability Considerations

### Horizontal Scaling

- **Stateless API**: All API instances are stateless
- **Load Balancing**: Multiple API instances behind load balancer
- **Database Sharding**: Shard by intersection or region
- **Redis Cluster**: Distributed caching

### Vertical Scaling

- **GPU Acceleration**: For ML model inference
- **Connection Pooling**: Efficient database connections
- **Caching Strategy**: Multi-level caching

### Performance Optimization

- **Async/Await**: Non-blocking I/O operations
- **Batch Processing**: Process multiple requests together
- **Model Optimization**: Quantization and pruning
- **CDN**: Static asset delivery

---

## Security Architecture

### Authentication & Authorization

- **OAuth2**: Token-based authentication
- **JWT**: Stateless tokens
- **Role-Based Access Control**: Fine-grained permissions

### Data Security

- **Encryption at Rest**: Database encryption
- **Encryption in Transit**: TLS/SSL
- **Secret Management**: Environment variables or vault

### API Security

- **Rate Limiting**: Prevent abuse
- **Input Validation**: Prevent injection attacks
- **CORS**: Cross-origin resource sharing control
- **HTTPS**: Enforced in production

---

## Deployment Architecture

### Development

```
Developer Machine
    └── Docker Compose
        ├── API Service
        ├── PostgreSQL
        └── Redis
```

### Production

```
Kubernetes Cluster
    ├── API Deployment (Replicas: 3)
    ├── Database StatefulSet
    ├── Redis Deployment
    ├── Dashboard Deployment
    └── Monitoring Stack
```

---

## Monitoring & Observability

### Metrics

- **Application Metrics**: Request rate, latency, errors
- **Business Metrics**: Wait times, throughput, efficiency
- **System Metrics**: CPU, memory, disk, network

### Logging

- **Structured Logging**: JSON format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Log Aggregation**: Centralized logging system

### Tracing

- **Distributed Tracing**: Track requests across services
- **Performance Profiling**: Identify bottlenecks

---

## Development Workflow

1. **Local Development**: Docker Compose setup
2. **Testing**: Unit, integration, and E2E tests
3. **CI/CD**: Automated testing and deployment
4. **Code Review**: Pull request process
5. **Deployment**: Staging → Production

---

## Future Enhancements

### Planned Features

- **Microservices Migration**: Break into smaller services
- **Event-Driven Architecture**: Event sourcing and CQRS
- **GraphQL Federation**: Distributed GraphQL
- **Edge Computing**: Deploy models at edge locations
- **Multi-Tenancy**: Support multiple organizations

---

## Additional Resources

- [Getting Started Guide](GETTING_STARTED.md)
- [API Documentation](../api/API_DOCUMENTATION.md)
- [Component Guides](COMPONENT_GUIDES.md)
- [Best Practices](BEST_PRACTICES.md)

---

**Last Updated**: November 30, 2024  
**Architecture Version**: 2.0.0

