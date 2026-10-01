# DynamicalAveragingNQS

Dynamical averaging of neural quantum states (NQS) in the Schrödinger and Heisenberg pictures.

## Installation

```bash
git clone --recurse-submodules https://github.com/rr637/DynamicalAveragingNQS.git
cd DynamicalAveragingNQS
bash conda_env_setup.sh        # NVIDIA GPU: bash conda_env_setup.sh gpu
conda activate da_nqs
```

`ptvmc-systematic-study` is a submodule (fork of [NeuralQXLab/ptvmc-systematic-study](https://github.com/NeuralQXLab/ptvmc-systematic-study)) used for p-tVMC. If you cloned without `--recurse-submodules`, run `git submodule update --init`.

## Schrödinger picture

Run in this order:

1. `VMC_Ising.py`: VMC for the 1D or 2D mixed-field Ising model (`RBMSymm` or `ViT`). Saves the log, with the parameters at every iteration, to `Results/<run>/`.
2. `p-tVMC.py` or `TDVP.py`: forward and backward real-time evolution of selected VMC iterations. Saves the trajectories to `Results/<run>/PTVMC_.../` or `Results/<run>/TDVP_.../`.
3. `resample_observables.py` (optional): re-evaluates the observables of the VMC states and of the time evolved states with more samples.
4. `NQS_DA_Ising.ipynb`: loads the VMC and time evolution results (the resampled logs if they exist) and computes the dynamically averaged observables and their errors for any window width sigma.

`DA_utils.py` contains the systems, models and loading functions used by all of the above; new systems and models are added there.

`Results/MixedIsing_L12_hx1.25_hz0.15_RBMSymm_alpha2_0914_1641` is the 1D run used in the paper and is loaded by default in `NQS_DA_Ising.ipynb`.

## Heisenberg picture

`NQS_DynamicalAveraging_HeisenbergCorrection2.ipynb`: self-contained notebook (written for Google Colab) that runs VMC for the 1D XYZ model and applies the Heisenberg picture dynamical averaging correction, computed from nested commutators of the Hamiltonian with each observable.
