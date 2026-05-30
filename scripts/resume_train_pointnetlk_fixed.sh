#!/bin/bash
# 从最佳权重继续训练 PointNetLK（使用修复后的数据集）
# 之前训练到 Epoch 107，最佳验证损失：0.003225
# 注意：需要手动修改 config 文件中的 checkpoint_dir

cd /home/linzhe/PCLR_compare
# Note: project reorganized - experiments/checkpoints/ is now experiments/checkpoints/
# common/ is now src/common/ 

# 激活环境
source activate PCLR_compare

echo "=========================================="
echo "继续训练 PointNetLK (从 Epoch 107)"
echo "使用修复后的数据集（返回 igt 正向变换）"
echo "=========================================="
echo ""
echo "⚠️  重要提示："
echo "   训练脚本不支持 --resume 参数"
echo "   需要你手动修改配置文件或训练代码"
echo ""
echo "建议方案："
echo "   1. 复制配置文件: cp src/benchmarking/bridges/configs/c3vd_pointnetlk.yaml src/benchmarking/bridges/configs/c3vd_pointnetlk_resume.yaml"
echo "   2. 修改新配置中的 checkpoint_dir 为: experiments/checkpoints/c3vd_pointnetlk_corrected"
echo "   3. 在训练代码中添加 resume 逻辑"
echo ""
echo "或者，我可以帮你从头开始训练，但使用修复后的数据集"
echo "=========================================="

# 注释掉原来的命令
# python src/benchmarking/bridges/train_pointnetlk_c3vd.py \
#   --config src/benchmarking/bridges/configs/c3vd_pointnetlk.yaml \
#   --stage pointnetlk \
#   --data-root /mnt/f/Datasets/C3VD_sever_datasets \
#   --resume experiments/checkpoints/c3vd_pointnetlk/c3vd_pointnetlk_model_snap_best.pth \
#   --checkpoint-dir experiments/checkpoints/c3vd_pointnetlk_corrected \
#   --epochs 300 \
#   2>&1 | tee experiments/logs/c3vd_pointnetlk_corrected/train_resume_$(date +%Y%m%d_%H%M%S).log
