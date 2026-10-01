import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_PLATFORM_NAME"] = "gpu"

import netket as nk
import numpy as np
import netket.experimental as nkx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt
import flax
import json
from flax import nnx
import optax

import os
print(f"NetKet version: {nk.__version__}")
from datetime import datetime
import netket as nk
from netket.operator.spin import sigmax, sigmaz, sigmay
from utils import get_1D_Ising_observables
import time
from models.FFN_Ising import FFNSymmModel
from models.RBM import RBMSymmModPhase
SAVING = True
date = datetime.now().strftime("%m%d_%H%M")


L = 12
hx = 1.25   # Transverse field strength (X direction)
hz = 0.15   # Longitudinal field strength (Z direction)
J  = 1.0    # Nearest-neighbor ZZ interaction strength
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)




alpha = 2
n_samples = 2**10
n_chains = 64
n_discard_per_chain = 16
lr = 7.5e-4
diag_shift = 5e-4     # optional: larger shift damps the minSR step -> extra smoothing
n_iters  = 3000      # need more iters since it converges slower (you're already at 500)

# Learning-rate cosine decay: lr_init -> lr_init * lr_alpha (smooth, low-variance plateau)
# lr_init  = 0.005      # was 0.02  -> ~4x slower, smoother transient
# lr_alpha = 0.3        # gentler decay (final lr = 0.0015), keeps it smooth throughout

# lr_schedule = optax.cosine_decay_schedule(
#     init_value=lr_init, decay_steps=n_iters, alpha=lr_alpha
# )
(ha,
corr_x, corr_x2, corr_x3,
X1, X2, X3, X4,
corr_z, corr_z2, corr_z3,
Z1, Z2, Z3, Z4,
avg_mag_x, avg_mag_z,
corr_x_avg, corr_z_avg) = get_1D_Ising_observables(hi, g, L, hx, hz, J)

# ma = RBMSymmModPhase(symmetries = g.translation_group(),alpha = alpha)
# ma = nk.models.RBMModPhase(alpha=alpha)
param_dtype = complex
use_visible_bias = True
use_hidden_bias = True
ma = nk.models.RBMSymm(
    symmetries=g.translation_group(),
    alpha=alpha,
    param_dtype=complex,
    use_visible_bias=use_visible_bias,
    use_hidden_bias=use_hidden_bias,
)



# ma = FFNSymmModel(L=L, alpha=4, rngs=nnx.Rngs(2))

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains, sweep_size=L)
# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
print("num params: ", vs.n_parameters)
# raise Exception



op = nk.optimizer.Sgd(learning_rate = lr)
sr = nk.optimizer.SR(diag_shift=diag_shift)
# Variational monte carlo driver
gs = nk.driver.VMC_SR(ha, op, variational_state=vs,    
    diag_shift=diag_shift,
    momentum=0,
    use_ntk=False)

def log_parameters(step, log_data, driver):
  # Callback frunctoin to store current NQS parameters in pars
  vs = driver.state   # get the VariationalState from the driver
  pars = flax.core.copy(vs.parameters, {})

  # Store inside runtime log dictionary
  log_data["Parameters"] = pars

  return True

# Full obs list added 4/8, previous only included for X1,Z1, and X2,Z2, avg_mag_x, avg_mag_z
obs = {
    "Energy": ha,
    "corr_x": corr_x,
    "corr_z": corr_z,
    "avg_mag_x": avg_mag_x,
    "avg_mag_z": avg_mag_z,
    "corr_x_avg": corr_x_avg,
    "corr_z_avg": corr_z_avg,
}

t0 = time.time()
log = nk.logging.RuntimeLog()
gs.run(n_iter=n_iters, out=log, obs=obs, callback=log_parameters)
t1 = time.time()
dir = f"Dynamic_Avg_Results3/MixedIsing_L{L}_hx{hx}_hz{hz}_{ma.__class__.__name__}_alpha{alpha}_{date}"

metadata = {


    "system": {
        "model_name": "1D mixed-field Ising model",
        "L": L,
        "graph_type": g.__class__.__name__,
        "n_dim": 1,
        "pbc": True,
        "hilbert_type": hi.__class__.__name__,
        "spin": 0.5,
        "N": g.n_nodes,
        "hx": hx,
        "hz": hz,
        "J": J,
    },

    "model": {
        "type": ma.__class__.__name__,
        "alpha": alpha,
        "param_dtype": "complex" if param_dtype == complex else "real",
        "use_visible_bias": use_visible_bias,
        "use_hidden_bias": use_hidden_bias,
        "n_parameters": vs.n_parameters

    },
    "sampler": {
        "type": sa.__class__.__name__,
        "n_chains": n_chains,
    },
    "variational_state": {
        "type": vs.__class__.__name__,
        "n_samples": n_samples,
        "n_discard_per_chain": n_discard_per_chain,

    },
    "optimizer": {"type": "SGD", 
                  "learning_rate": lr},
                #   "schedule": "cosine_decay",
                #   "lr_init": lr_init, "lr_alpha": lr_alpha, "momentum": 0},
    "sr": {
        "type": "SR",
        "diag_shift": diag_shift,
    },
    "vmc": {
        "n_iters": n_iters,
        "driver_type": gs.__class__.__name__,
        "runtime": t1 - t0,
        "observables": list(obs.keys()),

    },
}



if SAVING:
    os.makedirs(dir, exist_ok=True)
    log.serialize(f'{dir}/VMC_log.json')
    with open(f"{dir}/VMC_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)



