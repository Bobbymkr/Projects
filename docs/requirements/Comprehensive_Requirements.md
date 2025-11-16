# Adaptive Traffic Signal Control — Comprehensive Requirements

## Table of Contents
- [1) Project Overview](#1-project-overview-plain-language)
- [2) Visual Diagrams](#2-visual-diagrams)
- [3) User Stories](#3-user-stories-business-context--outcomes)
- [4) Plain-Language Technical Explanations](#4-plain-language-technical-explanations)
- [5) Inputs, Outputs, and Expected Behaviors](#5-inputs-outputs-and-expected-behaviors)
- [6) Glossary](#6-glossary)
- [7) Business Requirements](#7-business-requirements-stakeholders)
- [8) Technical Implementation](#8-technical-implementation-developers)

## 1) Project Overview (Plain Language)
- Purpose: Reduce traffic jams and wait times by making traffic lights smarter.
- How it works: The system watches roads (camera or simulator), predicts short-term traffic, and decides the next light change to keep cars moving.
- Goals:
  - Shorten average wait times (target: 30–50% reduction).
  - Reduce queue lengths (target: 25–40% reduction).
  - Keep intersections safe with reliable, rule-based timing.
  - Provide clear dashboards for performance and health.
- See business goals in [Section 7](#7-business-requirements-stakeholders).

## 2) Visual Diagrams
- Architecture Overview: High-level parts and how they connect  
  ![Architecture Overview](../diagrams/architecture_overview.svg)
- System Flowchart: End-to-end process from inputs to outcomes  
  ![System Flowchart](../diagrams/system_flowchart.svg)
 - Roles & Responsibilities: Who uses the system and who builds it  
  ![Roles & Responsibilities](../diagrams/roles_responsibilities.svg)

## 3) User Stories (Business Context + Outcomes)
- As a traffic manager, I want to choose a scenario (e.g., morning rush) so the system uses suitable timing to reduce congestion.
- As a city operator, I want live monitoring of wait times and queues so I can see benefits and catch issues early.
- As a safety officer, I want minimum green/red rules enforced so signals remain safe under all conditions.
- As a data analyst, I want reports comparing methods (AI vs. traditional) so I can justify investment decisions.
- As a maintenance staff, I want simple setup tools on Windows so the system can be installed and updated quickly.
- As a citizen, I want shorter waits at lights so travel becomes faster and more predictable.
- As a planner, I want simulation-based trials so changes can be tested before affecting real roads.
- As a compliance officer, I want secure logging without personal data so we meet privacy requirements.

## 4) Plain-Language Technical Explanations
- Cameras & Counting: Video is processed to detect cars and estimate how many are waiting per lane.
- Traffic Simulator: A virtual city where cars arrive and move; used to test strategies safely and quickly.
- AI Brain (Reinforcement Learning): Learns good timing patterns over time by trying actions and seeing results.
- Forecasting: Short-term predictions of how busy each approach will be; helps the AI choose better actions.
- Helpful Rules: Traditional methods (like Webster’s) and fuzzy logic support safe, sensible timings.
- Decision Hub: Combines data from cameras/simulator, forecasts, and rules to select the next light change.
- Controller: Applies the chosen timing to the signal safely and consistently.
- Monitoring: Collects metrics (wait time, queue length, throughput) and shows them in dashboards.

## 5) Inputs, Outputs, and Expected Behaviors
- Inputs:
  - Video stream (camera or file) showing an intersection.
  - Simulation scenarios (e.g., morning rush, balanced flow).
  - Configuration (minimum/maximum green times, lanes, arrival rates).
- Outputs:
  - Next signal action (e.g., extend green for eastbound by 5 seconds).
  - Metrics (average wait, queue length, throughput).
  - Reports and logs for auditing and performance.
- Expected Behaviors:
  - Always respect safety constraints (min green/red).
  - Prefer actions that reduce queues and waits.
  - Recover gracefully from input errors (invalid config → use safe defaults).
  - Provide clear health checks and monitoring data.

## 6) Glossary
- Intersection: Where roads cross with traffic lights.
- Queue: Number of cars waiting at a light.
- Throughput: How many cars pass through in a period.
- Scenario: Predefined traffic pattern for testing (e.g., morning rush).
- Reinforcement Learning (RL): AI method that learns by trial and feedback.
- DQN (Deep Q-Network): A common RL approach using neural networks.
- Forecasting: Predicting near-future traffic volumes.
- Simulator (SUMO): A virtual model of roads and traffic behavior.
- YOLO: Computer vision model that detects objects (cars) in images.
- Policy: AI’s strategy for choosing actions.
- Episode: One run of simulation used for training/testing.
- Metric: A number that measures performance (wait time, queue length).

## 7) Business Requirements (Stakeholders)
- Objectives:
  - Reduce average intersection wait time by at least 30%.
  - Reduce queue lengths by at least 25%.
  - Maintain 99.0% system uptime during scheduled hours.
- Constraints:
  - Operates on Windows (primary), Python 3.9+.
  - No personal data stored in logs; privacy preserved.
  - Must enforce safety timing rules at all times.
- KPIs:
  - Wait Time, Queue Length, Throughput, Uptime.
  - Incident Rate (unexpected signal errors).
- Acceptance Criteria:
  - Demonstrated improvement vs. traditional method in simulation.
  - Clear, readable dashboards of key metrics.
  - Documented safe fallback behavior on invalid inputs.
- Reporting:
  - Weekly performance summaries.
  - Algorithm comparison reports for decision-makers.

## 8) Technical Implementation (Developers)
- Components (code folders):
  - RL & Decisions: `src/rl/*` (training, inference, benchmarking)
  - Environments: `src/env/*` (traffic, SUMO, video, multi-agent)
  - Vision: `src/vision/*` (video pipeline, detection, ROI)
  - Forecasting: `src/forecast/*`
  - Controllers: `src/control/*` (Fuzzy, Webster, GA/PSO)
  - Utilities & Security: `src/utils/*`, `src/security/*`
  - Scenarios: `configs/*`
- Data Flow:
  - Perception/environment → decision hub → controller → monitoring.
- Interfaces:
  - Config loader: reads JSON scenario/settings; validates and provides defaults.
  - Inference API: receives state/vision inputs; outputs next action.
  - Metrics API: records counters and timings; exposes dashboards/logs.
- Safety & Error Handling:
  - Input validation on configs and video sources.
  - Enforce min/max green per phase.
  - Structured exceptions with secure logging (no sensitive data).
- Performance:
  - Target sub-second decisions; batch vision processing where possible.
  - Use GPU optional for training; CPU acceptable for inference.
- Testing:
  - Unit/integration/system tests (pytest).
  - Performance benchmarks vs. traditional controllers.