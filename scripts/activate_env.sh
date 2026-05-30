#!/bin/bash
# Activate PCLR_compare environment and set up Python path

echo "================================================"
echo "Activating PCLR_compare Environment"
echo "================================================"

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate PCLR_compare

# Set Python path
export PYTHONPATH=/home/linzhe/PCLR_compare:$PYTHONPATH

# Change to project directory
cd /home/linzhe/PCLR_compare

# Show environment info
echo ""
echo "Environment: PCLR_compare"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "Working directory: $(pwd)"
echo ""
echo "================================================"
echo "Environment activated! You can now run:"
echo ""
echo "Verify environment:"
echo "  python verify_env.py"
echo ""
echo "Test dataset loading:"
echo "  python src/common/test_dataset.py --data-root /path/to/C3VD_datasets"
echo ""
echo "Train DCP:"
echo "  python src/benchmarking/bridges/train_dcp_c3vd.py --config src/benchmarking/bridges/configs/c3vd_dcp.yaml"
echo ""
echo "Train RegTR (if dependencies installed):"
echo "  python src/benchmarking/bridges/train_regtr_c3vd.py --config src/benchmarking/bridges/configs/c3vd_regtr.yaml --dev"
echo ""
echo "================================================"
echo ""
