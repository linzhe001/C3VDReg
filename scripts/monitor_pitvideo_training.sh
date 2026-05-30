#!/bin/bash

# 监控PITvideo训练进度的脚本

echo "=== PITvideo Mamba3D Training Monitor ==="
echo ""

# 检查进程
echo "1. Checking training process..."
if ps aux | grep -v grep | grep "train_mamba3d_pitvideo" > /dev/null; then
    echo "   ✓ Training process is RUNNING"
    ps aux | grep -v grep | grep "train_mamba3d_pitvideo" | awk '{print "   PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "   ✗ No training process found"
fi
echo ""

# 最新日志文件
echo "2. Latest log file:"
LATEST_LOG=$(ls -t experiments/logs/pitvideo_mamba3d_finetune/train_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "   $LATEST_LOG"
    echo ""
    echo "3. Recent training progress (last 30 lines):"
    tail -30 "$LATEST_LOG" | grep -E "(Epoch|train_loss|val_loss|completed|ERROR)"
else
    echo "   No log files found"
fi
echo ""

# 检查点
echo "4. Saved checkpoints:"
ls -lh experiments/checkpoints/pitvideo_mamba3d_finetune/*.pth 2>/dev/null | awk '{print "   "$9, $5}'
echo ""

echo "=== End of Monitor ==="
