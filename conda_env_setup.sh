#!/bin/bash
# Create the conda env for this repository
# Laptop (CPU):     bash conda_env_setup.sh
# NVIDIA GPU:       bash conda_env_setup.sh gpu
# Then activate it: conda activate da_nqs

set -e

ENV_NAME=da_nqs
DEVICE=${1:-cpu}

eval "$(conda shell.bash hook)"

conda create -n $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

pip install numpy scipy matplotlib jupyter
if [ "$DEVICE" == "gpu" ]; then
    JAX="jax[cuda12]==0.11.1"
else
    JAX="jax==0.11.1"
fi
# One pip command so that jax, jaxlib and the CUDA plugin stay at the same version
pip install "$JAX" flax==0.12.10 optax==0.2.8 netket==3.23.0 plum-dispatch==2.9.0 beartype==0.22.9 einops nqxpack

# Verify
python -c "
import jax
import netket as nk
print(f'JAX {jax.__version__}')
print(f'NetKet {nk.__version__}')
print(f'Backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')
"

# Optional check: is the ptvmc submodule there and importable? (only needed for p-tVMC.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if PYTHONPATH="$SCRIPT_DIR/ptvmc-systematic-study/packages" \
   python -c "import ptvmc, netket_pro, advanced_drivers" 2>/dev/null; then
    echo "ptvmc submodule: OK"
else
    echo "ptvmc submodule: not found (only needed for p-tVMC.py; get it with: git submodule update --init)"
fi


echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
