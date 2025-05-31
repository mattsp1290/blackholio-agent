# How to Train Your Blackholio ML Agent

## Project Status ✅

**The Blackholio ML Agent project is FULLY COMPLETED** as of May 25, 2025. All 10 major components are finished:

- ✅ Environment wrapper
- ✅ ML model architecture  
- ✅ Training pipeline
- ✅ Inference system
- ✅ Training executable script
- ✅ Testing and validation
- ✅ Documentation
- ✅ Advanced features (multi-agent, self-play)

## Prerequisites

Before starting training, ensure you have:

1. **Blackholio game server running** on `localhost:3000`
2. You're in the project directory: `/Users/punk1290/git/blackholio-agent`
3. All dependencies installed (see `requirements.txt`)
4. Sufficient disk space for logs and checkpoints (~1-2GB for 6-hour training)

## Quick Verification Test (5 minutes)

**ALWAYS run this verification test before committing to a long training session:**

```bash
python scripts/train_agent.py \
  --total-timesteps 1000 \
  --n-envs 2 \
  --experiment-name verification_test
```

This 30-second test will:
- ✅ Validate server connection to Blackholio
- ✅ Initialize the training environment  
- ✅ Run a few training steps
- ✅ Confirm all systems work correctly

**Expected output:**
```
🚀 Starting Blackholio RL Agent Training
✅ Successfully connected to Blackholio server at localhost:3000
✅ Configuration validation passed
🎯 Starting training...
```

If you see any ❌ errors, troubleshoot before proceeding to full training.

## 6-Hour Training Session

### Main Training Command

```bash
python scripts/train_agent.py \
  --config scripts/configs/quick_train.yaml \
  --total-timesteps 3000000 \
  --experiment-name 6hour_training_run \
  --n-envs 8
```

### What This Does

- **Duration**: ~6 hours (3M timesteps with 8 parallel environments)
- **Auto-saves**: Checkpoints every 20 minutes
- **Logs**: Stored in `logs/6hour_training_run/`
- **Models**: Saved in `checkpoints/6hour_training_run/`
- **Curriculum**: Progresses through food collection → survival → combat → advanced strategies

### Training Progress

You'll see real-time output like:
```
📊 Episode: 1250 | Reward: 45.2 | Loss: 0.023 | ETA: 4h 32m
🎯 Curriculum Stage: survival (60% complete)
💾 Checkpoint saved: latest_model.pth
```

## Real-Time Monitoring

### TensorBoard (Recommended)

Open a **second terminal** and run:

```bash
tensorboard --logdir logs/6hour_training_run/tensorboard
```

Then visit: **http://localhost:6006**

You'll see live graphs of:
- Training reward curves
- Loss functions
- Episode statistics
- Learning progress

### Console Monitoring

The training script provides live updates every 10 steps:
- Current reward scores
- Training loss values
- Estimated time remaining
- Curriculum learning progress

## Safety Features

### Graceful Stop
- Press **Ctrl+C** to safely stop training
- Current model will be saved automatically
- No data loss

### Auto-Resume
If training is interrupted, resume with:
```bash
python scripts/train_agent.py \
  --resume checkpoints/6hour_training_run/latest_model.pth
```

### Automatic Checkpoints
- Models saved every 20 minutes
- Best performing models kept
- Logs continuously written to disk

## Monitoring Your Training

### Key Metrics to Watch

1. **Episode Reward**: Should generally increase over time
2. **Policy Loss**: Should decrease and stabilize
3. **Value Loss**: Should decrease over time
4. **Survival Time**: Agent should live longer as it learns

### Healthy Training Signs
- ✅ Rewards trending upward (with normal fluctuations)
- ✅ Agent surviving longer in later episodes
- ✅ Curriculum stages progressing automatically
- ✅ No connection errors to Blackholio server

### Warning Signs
- ❌ Rewards consistently decreasing
- ❌ Frequent server disconnections
- ❌ Very high or NaN loss values
- ❌ Agent dying immediately every episode

## Configuration Options

### Quick Training (1-2 hours)
```bash
python scripts/train_agent.py --config scripts/configs/quick_train.yaml
```

### Full Self-Play Training (8-12 hours)
```bash
python scripts/train_agent.py --config scripts/configs/full_train_selfplay.yaml
```

### Custom Training
Modify parameters directly:
```bash
python scripts/train_agent.py \
  --total-timesteps 5000000 \
  --learning-rate 0.0003 \
  --n-envs 16 \
  --batch-size 512
```

## After Training

### Test Your Trained Agent

```bash
python scripts/run_agent.py \
  --model checkpoints/6hour_training_run/best_model.pth \
  --host localhost:3000
```

### Analyze Results

Check the training notebook:
```bash
jupyter notebook examples/notebooks/training_visualization.ipynb
```

## Troubleshooting

### "Cannot connect to Blackholio server"
1. Ensure Blackholio is running: `./start_blackholio.sh`
2. Check if port 3000 is available
3. Try: `--skip-server-check` flag to bypass validation

### "CUDA/MPS not available"
- Training will automatically fall back to CPU
- For faster training, ensure GPU drivers are installed

### "Out of memory"
- Reduce `--n-envs` (try 4 instead of 8)
- Reduce `--batch-size` (try 128 instead of 256)

### Training stuck/not improving
- Check TensorBoard for loss curves
- Ensure Blackholio server is responsive
- Try different learning rates (0.0001 to 0.001)

### Server disconnections
- Check network connectivity
- Restart Blackholio server: `./stop_blackholio.sh && ./start_blackholio.sh`
- Use `--resume` to continue from last checkpoint

## Advanced Options

### Multi-Agent Training
```bash
python examples/multi_agent_demo.py
```

### Custom Rewards
Edit `src/blackholio_agent/environment/reward_calculator.py`

### Hyperparameter Tuning
See `docs/TRAINING_GUIDE.md` for detailed parameter explanations

## Getting Help

- **Training Guide**: `docs/TRAINING_GUIDE.md`
- **API Documentation**: `docs/API.md`
- **Test Results**: Run `python scripts/run_tests.py`
- **Performance Benchmarks**: Check `src/blackholio_agent/tests/performance/`

---

## Quick Reference Commands

```bash
# Verify everything works (30 seconds)
python scripts/train_agent.py --total-timesteps 1000 --n-envs 2 --experiment-name test

# 6-hour training session
python scripts/train_agent.py --config scripts/configs/quick_train.yaml --total-timesteps 3000000 --experiment-name 6hour_run

# Monitor with TensorBoard
tensorboard --logdir logs/6hour_run/tensorboard

# Resume interrupted training
python scripts/train_agent.py --resume checkpoints/6hour_run/latest_model.pth

# Test trained agent
python scripts/run_agent.py --model checkpoints/6hour_run/best_model.pth
```

**Happy Training! 🚀**
