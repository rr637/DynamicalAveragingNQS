import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["NETKET_SHARDING"] = "0"

import json
import time
from datetime import datetime

import jax
import flax
import netket as nk

from utils import build_system, build_model

print(f"NetKet version: {nk.__version__}")
print(f"Devices: {jax.devices()}")

SAVING = True
date = datetime.now().strftime("%m%d_%H%M")


# ============================================================
# System
# ============================================================
system = "1D"   # "1D" or "2D" mixed-field Ising model

if system == "1D":
    L = 12
    hx = 1.25   # Transverse field strength (X direction)
    hz = 0.15   # Longitudinal field strength (Z direction)
    J = 1.0     # Nearest-neighbor ZZ interaction strength

elif system == "2D":
    L = 6       # L x L lattice
    hx = 3.3
    hz = 0.05
    J = 1.0


# ============================================================
# Model and VMC settings
# ============================================================
model = "RBMSymm"   # "RBMSymm" or "ViT" (2D only)

if model == "RBMSymm":
    alpha = 2
    n_samples = 2**10
    n_chains = 64
    n_discard_per_chain = 16
    sweep_size = None     # None -> number of spins
    lr = 7.5e-4
    diag_shift = 5e-4
    momentum = 0
    use_ntk = False
    n_iters = 3000

elif model == "ViT":
    num_layers = 2
    d_model = 48
    heads = 8
    b = 2                 # patch size, L must be divisible by b
    expansion_factor = 2
    transl_invariant = True
    n_samples = 2**11
    n_chains = 64
    n_discard_per_chain = 16
    sweep_size = L**2 // 2
    lr = 1e-3
    diag_shift = 5e-4
    momentum = None
    use_ntk = True        # minSR, needed for the large number of ViT parameters
    n_iters = 2000


# ============================================================
# Build system and model
# ============================================================
metadata = {
    "system": {
        "model_name": f"{system} mixed-field Ising model",
        "L": L,
        "n_dim": 1 if system == "1D" else 2,
        "pbc": True,
        "hx": hx,
        "hz": hz,
        "J": J,
    },
}

if model == "RBMSymm":
    metadata["model"] = {
        "type": "RBMSymm",
        "alpha": alpha,
        "param_dtype": "complex",
        "use_visible_bias": True,
        "use_hidden_bias": True,
    }
    model_tag = f"RBMSymm_alpha{alpha}"

elif model == "ViT":
    metadata["model"] = {
        "type": "ViT",
        "num_layers": num_layers,
        "d_model": d_model,
        "n_heads": heads,
        "patch_size": b,
        "expansion_factor": expansion_factor,
        "transl_invariant": transl_invariant,
    }
    model_tag = f"ViT_L{num_layers}_d{d_model}"

g, hi, ha, obs = build_system(metadata["system"])
ma = build_model(metadata, g)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, sweep_size=sweep_size)
# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
print("num params: ", vs.n_parameters)


# ============================================================
# VMC
# ============================================================
op = nk.optimizer.Sgd(learning_rate=lr)
# Variational monte carlo driver
gs = nk.driver.VMC_SR(
    ha, op, variational_state=vs,
    diag_shift=diag_shift,
    momentum=momentum,
    use_ntk=use_ntk,
)

def log_parameters(step, log_data, driver):
    # Callback function to store current NQS parameters in the log
    log_data["Parameters"] = flax.core.copy(driver.state.parameters, {})
    return True

# Run VMC
t0 = time.time()
log = nk.logging.RuntimeLog()
gs.run(n_iter=n_iters, out=log, obs=obs, callback=log_parameters)
t1 = time.time()


# ============================================================
# Save
# ============================================================
prefix = "MixedIsing" if system == "1D" else "2DMixedIsing"
VMC_dir = f"Results/{prefix}_L{L}_hx{hx}_hz{hz}_{model_tag}_{date}"

metadata["model"]["n_parameters"] = vs.n_parameters
metadata["sampler"] = {
    "type": "MetropolisLocal",
    "n_chains": n_chains,
    "sweep_size": sa.sweep_size,
}
metadata["variational_state"] = {
    "type": "MCState",
    "n_samples": n_samples,
    "n_discard_per_chain": n_discard_per_chain,
}
metadata["optimizer"] = {
    "type": "SGD",
    "learning_rate": lr,
}
metadata["sr"] = {
    "type": "minSR" if use_ntk else "SR",
    "diag_shift": diag_shift,
    "momentum": momentum,
}
metadata["vmc"] = {
    "n_iters": n_iters,
    "driver_type": gs.__class__.__name__,
    "runtime": t1 - t0,
    "observables": list(obs.keys()),
}

if SAVING:
    os.makedirs(VMC_dir, exist_ok=True)
    log.serialize(f"{VMC_dir}/VMC_log.json")
    with open(f"{VMC_dir}/VMC_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved VMC results under: {VMC_dir}")
