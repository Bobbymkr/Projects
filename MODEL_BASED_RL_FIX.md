# Model-Based RL World Model Training Fix

## Issue
The "World model not trained. Returning random action." warnings were appearing repeatedly (hundreds of times) before the world model was trained.

## Root Cause
1. World model needs to collect transitions first (minimum 30-50)
2. Training only happened every 25 transitions
3. Warnings were logged every single time `select_action` was called
4. Training wasn't aggressive enough early on

## Fixes Applied

### 1. Reduced Warning Noise
- Changed warning to only log **once** instead of every call
- Added `_warning_logged` flag to track if warning was shown

### 2. Faster Training Initiation
- Reduced `min_transitions_for_training` from 50 to **30**
- Reduced `training_interval` from 25 to **20**
- Training happens immediately when threshold is reached

### 3. More Aggressive Training
- Train every **2 episodes** for Model-Based RL (instead of 10)
- Train immediately when enough transitions are collected
- Train with more epochs (10-20) based on data size

### 4. Automatic Training in select_action
- World model trains automatically in `select_action()` if enough data exists
- Prevents using untrained model unnecessarily

## Result
- ✅ Warnings appear only once
- ✅ World model trains faster (after 30 transitions)
- ✅ Training happens more frequently
- ✅ Better performance after training

## Testing
Run training and you should see:
1. One warning: "World model not trained yet. Using random actions until enough data is collected."
2. Training messages: "Training World Model on X transitions"
3. Success message: "World model trained on X transitions - now using MPC"
4. No more warnings after training

