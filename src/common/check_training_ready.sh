#!/bin/bash
#
# Check training prerequisites.
# Validate config files, dataset paths, training scripts, and runtime dependencies.
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WORK_DIR="/home/linzhe/PCLR_compare"
cd "$WORK_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Checking C3VD training readiness${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

ERRORS=0
WARNINGS=0

#==============================================================================
# 1. Check config files
#==============================================================================
echo -e "${YELLOW}[1/6] Checking config files...${NC}"

CONFIGS=(
    "src/benchmarking/bridges/configs/c3vd_pointnetlk_revisited.yaml"
    "src/benchmarking/bridges/configs/c3vd_pointnetlk.yaml"
    "src/benchmarking/bridges/configs/c3vd_dcp.yaml"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        echo -e "  ${GREEN}✓${NC} $config"
    else
        echo -e "  ${RED}✗${NC} $config (missing)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

#==============================================================================
# 2. Check training scripts
#==============================================================================
echo -e "${YELLOW}[2/6] Checking training scripts...${NC}"

SCRIPTS=(
    "src/benchmarking/bridges/train_pointnetlk_revisited_c3vd.py"
    "src/benchmarking/bridges/train_pointnetlk_c3vd.py"
    "src/benchmarking/bridges/train_dcp_c3vd.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "  ${GREEN}✓${NC} $script"
    else
        echo -e "  ${RED}✗${NC} $script (missing)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

#==============================================================================
# 3. Check dataset path
#==============================================================================
echo -e "${YELLOW}[3/6] Checking dataset path...${NC}"

# Extract data_root from config files.
DATA_ROOT=$(grep -h "data_root:" src/benchmarking/bridges/configs/c3vd_*.yaml | head -1 | cut -d':' -f2 | xargs)

if [ -z "$DATA_ROOT" ]; then
    echo -e "  ${RED}✗${NC} Could not read data_root from config files"
    ERRORS=$((ERRORS + 1))
elif [ ! -d "$DATA_ROOT" ]; then
    echo -e "  ${RED}✗${NC} Dataset directory does not exist: $DATA_ROOT"
    ERRORS=$((ERRORS + 1))
else
    echo -e "  ${GREEN}✓${NC} Dataset directory: $DATA_ROOT"

    # Check subdirectories.
    if [ -d "$DATA_ROOT/C3VD_ply_source" ]; then
        SOURCE_COUNT=$(find "$DATA_ROOT/C3VD_ply_source" -name "*.ply" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} C3VD_ply_source/ (${SOURCE_COUNT} .ply files)"
    else
        echo -e "  ${RED}✗${NC} C3VD_ply_source/ subdirectory is missing"
        ERRORS=$((ERRORS + 1))
    fi

    if [ -d "$DATA_ROOT/visible_point_cloud_ply_depth" ]; then
        TARGET_COUNT=$(find "$DATA_ROOT/visible_point_cloud_ply_depth" -name "*.ply" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} visible_point_cloud_ply_depth/ (${TARGET_COUNT} .ply files)"
    else
        echo -e "  ${RED}✗${NC} visible_point_cloud_ply_depth/ subdirectory is missing"
        ERRORS=$((ERRORS + 1))
    fi
fi
echo ""

#==============================================================================
# 4. Check checkpoint directories
#==============================================================================
echo -e "${YELLOW}[4/6] Checking checkpoint directories...${NC}"

CKPT_BASE="/home/linzhe/PCLR_compare/experiments/checkpoints"

if [ ! -d "$CKPT_BASE" ]; then
    echo -e "  ${YELLOW}⚠${NC} Checkpoint base directory does not exist and will be created: $CKPT_BASE"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "  ${GREEN}✓${NC} Checkpoint base directory: $CKPT_BASE"
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
            echo -e "  ${YELLOW}⚠${NC} $dir (${COUNT} existing checkpoints may be overwritten)"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "  ${GREEN}✓${NC} $dir (empty directory)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} $dir (will be created)"
        WARNINGS=$((WARNINGS + 1))
    fi
done
echo ""

#==============================================================================
# 5. Check Python environment
#==============================================================================
echo -e "${YELLOW}[5/6] Checking Python environment...${NC}"

# Activate conda environment.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PCLR_compare 2>/dev/null || {
    echo -e "  ${RED}✗${NC} Could not activate conda environment: PCLR_compare"
    ERRORS=$((ERRORS + 1))
}

# Check Python version.
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo -e "  ${GREEN}✓${NC} Python version: $PYTHON_VERSION"

# Check required packages.
PACKAGES=("torch" "numpy" "yaml" "tqdm")

for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo -e "  ${GREEN}✓${NC} $pkg ($VERSION)"
    else
        echo -e "  ${RED}✗${NC} $pkg (not installed)"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

#==============================================================================
# 6. Check GPU
#==============================================================================
echo -e "${YELLOW}[6/6] Checking GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "  ${GREEN}✓${NC} Detected $GPU_COUNT GPU(s):"

    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while IFS=',' read -r idx name mem_total mem_free; do
        echo -e "    GPU $idx: $name"
        echo -e "           Memory: $mem_free / $mem_total"
    done
else
    echo -e "  ${RED}✗${NC} No GPU detected (nvidia-smi unavailable)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

#==============================================================================
# Summary
#==============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Readiness check complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed. Training can start.${NC}"
    echo ""
    echo -e "${BLUE}Training command:${NC}"
    echo -e "  python scripts/runners/train_benchmark.py --config configs/benchmark/train_r90_t500mm_0_200epoch/train_geotransformer.yaml"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Found $WARNINGS warning(s), but training can continue.${NC}"
    echo ""
    echo -e "${BLUE}Training command:${NC}"
    echo -e "  python scripts/runners/train_benchmark.py --config configs/benchmark/train_r90_t500mm_0_200epoch/train_geotransformer.yaml"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo -e "${RED}Resolve the errors before starting training.${NC}"
    echo ""
    exit 1
fi
