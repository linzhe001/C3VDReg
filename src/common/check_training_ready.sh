#!/bin/bash
#
# 检查训练前的准备工作
# 验证配置文件、数据集、训练脚本等
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WORK_DIR="/home/linzhe/PCLR_compare"
cd "$WORK_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}检查C3VD训练准备情况${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

ERRORS=0
WARNINGS=0

#==============================================================================
# 1. 检查配置文件
#==============================================================================
echo -e "${YELLOW}[1/6] 检查配置文件...${NC}"

CONFIGS=(
    "src/benchmarking/bridges/configs/c3vd_pointnetlk_revisited.yaml"
    "src/benchmarking/bridges/configs/c3vd_pointnetlk.yaml"
    "src/benchmarking/bridges/configs/c3vd_dcp.yaml"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        echo -e "  ${GREEN}✓${NC} $config"
    else
        echo -e "  ${RED}✗${NC} $config (缺失)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

#==============================================================================
# 2. 检查训练脚本
#==============================================================================
echo -e "${YELLOW}[2/6] 检查训练脚本...${NC}"

SCRIPTS=(
    "src/benchmarking/bridges/train_pointnetlk_revisited_c3vd.py"
    "src/benchmarking/bridges/train_pointnetlk_c3vd.py"
    "src/benchmarking/bridges/train_dcp_c3vd.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script"
    else
        echo -e "  ${RED}✗${NC} $script (缺失)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

#==============================================================================
# 3. 检查数据集路径
#==============================================================================
echo -e "${YELLOW}[3/6] 检查数据集路径...${NC}"

# 从配置文件中提取data_root
DATA_ROOT=$(grep -h "data_root:" src/benchmarking/bridges/configs/c3vd_*.yaml | head -1 | cut -d':' -f2 | xargs)

if [ -z "$DATA_ROOT" ]; then
    echo -e "  ${RED}✗${NC} 无法从配置文件读取data_root"
    ERRORS=$((ERRORS + 1))
elif [ ! -d "$DATA_ROOT" ]; then
    echo -e "  ${RED}✗${NC} 数据集目录不存在: $DATA_ROOT"
    ERRORS=$((ERRORS + 1))
else
    echo -e "  ${GREEN}✓${NC} 数据集目录: $DATA_ROOT"

    # 检查子目录
    if [ -d "$DATA_ROOT/C3VD_ply_source" ]; then
        SOURCE_COUNT=$(find "$DATA_ROOT/C3VD_ply_source" -name "*.ply" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} C3VD_ply_source/ (${SOURCE_COUNT} 个.ply文件)"
    else
        echo -e "  ${RED}✗${NC} C3VD_ply_source/ 子目录缺失"
        ERRORS=$((ERRORS + 1))
    fi

    if [ -d "$DATA_ROOT/visible_point_cloud_ply_depth" ]; then
        TARGET_COUNT=$(find "$DATA_ROOT/visible_point_cloud_ply_depth" -name "*.ply" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} visible_point_cloud_ply_depth/ (${TARGET_COUNT} 个.ply文件)"
    else
        echo -e "  ${RED}✗${NC} visible_point_cloud_ply_depth/ 子目录缺失"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

#==============================================================================
# 4. 检查Checkpoint目录
#==============================================================================
echo -e "${YELLOW}[4/6] 检查Checkpoint目录...${NC}"

CKPT_BASE="/home/linzhe/PCLR_compare/experiments/checkpoints"

if [ ! -d "$CKPT_BASE" ]; then
    echo -e "  ${YELLOW}⚠${NC} Checkpoint基础目录不存在，将自动创建: $CKPT_BASE"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "  ${GREEN}✓${NC} Checkpoint基础目录: $CKPT_BASE"
fi

CKPT_DIRS=(
    "$CKPT_BASE/c3vd_pointnetlk_revisited"
    "$CKPT_BASE/c3vd_pointnetlk"
    "$CKPT_BASE/c3vd_dcp"
)

for dir in "${CKPT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        COUNT=$(ls -1 "$dir"/*.pth 2>/dev/null | wc -l)
        if [ $COUNT -gt 0 ]; then
            echo -e "  ${YELLOW}⚠${NC} $dir (已有${COUNT}个checkpoint，将被覆盖)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "  ${GREEN}✓${NC} $dir (空目录)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} $dir (将自动创建)"
        WARNINGS=$((WARNINGS + 1))
    fi
done
echo ""

#==============================================================================
# 5. 检查Python环境
#==============================================================================
echo -e "${YELLOW}[5/6] 检查Python环境...${NC}"

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PCLR_compare 2>/dev/null || {
    echo -e "  ${RED}✗${NC} 无法激活conda环境: PCLR_compare"
    ERRORS=$((ERRORS + 1))
}

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo -e "  ${GREEN}✓${NC} Python版本: $PYTHON_VERSION"

# 检查关键包
PACKAGES=("torch" "numpy" "yaml" "tqdm")

for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "未知")
        echo -e "  ${GREEN}✓${NC} $pkg ($VERSION)"
    else
        echo -e "  ${RED}✗${NC} $pkg (未安装)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

#==============================================================================
# 6. 检查GPU
#==============================================================================
echo -e "${YELLOW}[6/6] 检查GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "  ${GREEN}✓${NC} 检测到 $GPU_COUNT 个GPU:"

    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while IFS=',' read -r idx name mem_total mem_free; do
        echo -e "    GPU $idx: $name"
        echo -e "           内存: $mem_free / $mem_total"
    done
else
    echo -e "  ${RED}✗${NC} 未检测到GPU (nvidia-smi不可用)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

#==============================================================================
# 总结
#==============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}检查完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！可以开始训练。${NC}"
    echo ""
    echo -e "${BLUE}启动训练命令:${NC}"
    echo -e "  python scripts/benchmark/launch_train_rollout.py --audit outputs/benchmark/train_rollout_audit/train_rollout_audit.json --ready-only"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ 发现 $WARNINGS 个警告，但可以继续训练。${NC}"
    echo ""
    echo -e "${BLUE}启动训练命令:${NC}"
    echo -e "  python scripts/benchmark/launch_train_rollout.py --audit outputs/benchmark/train_rollout_audit/train_rollout_audit.json --ready-only"
    echo ""
    exit 0
else
    echo -e "${RED}✗ 发现 $ERRORS 个错误，$WARNINGS 个警告。${NC}"
    echo -e "${RED}请解决错误后再开始训练。${NC}"
    echo ""
    exit 1
fi
