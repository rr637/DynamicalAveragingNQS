#!/bin/bash
# Create the conda env for this repository
# Laptop (CPU):     bash conda_env_setup.sh
# NVIDIA GPU:       bash conda_env_setup.sh gpu
# Then activate it: conda activate da_nqs

set -e

ENV_NAME=da_nqs
DEVICE=${1:-cpu}

eval "$(conda shell.bash hook)"

conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

pip install numpy scipy matplotlib jupyter
if [ "$DEVICE" == "gpu" ]; then
    pip install "jax[cuda12]==0.9.2"
else
    pip install "jax==0.9.2"
fi
pip install flax==0.12.6 optax==0.2.8 netket==3.21.0
pip install einops nqxpack

# Verify
python -c "
import jax
import netket as nk
print(f'JAX {jax.__version__}')
print(f'NetKet {nk.__version__}')
print(f'Backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')
"

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
