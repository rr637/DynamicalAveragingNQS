import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "0"

import json
import time
from datetime import datetime

import jax
import flax
import numpy as np
import netket as nk
import netket.experimental as nkx

from utils import build_system, build_model, params_at_iter

print(f"JAX {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")


def log_parameters(step, log_data, driver):
    # Callback function to store current NQS parameters in the log
    log_data["Parameters"] = flax.core.copy(driver.state.parameters, {})
    return True


# ============================================================
# User settings
# ============================================================
SAVING = True
FORWARD = True
BACKWARD = True

VMC_dir = "Results/MixedIsing_L12_hx1.25_hz0.15_RBMSymm_alpha2_0914_1641"


# ============================================================
# No VMC results yet: uncomment to run a short VMC here, save it as
# a VMC folder in Results/ and time evolve its states
# ============================================================
# vmc_metadata = {
#     "system": {"model_name": "1D mixed-field Ising model", "L": 12, "n_dim": 1, "pbc": True, "hx": 1.25, "hz": 0.15, "J": 1.0},
#     "model": {"type": "RBMSymm", "alpha": 2, "param_dtype": "complex", "use_visible_bias": True, "use_hidden_bias": True},
#     "vmc": {"n_iters": 300},
# }
# g, hi, ha, obs = build_system(vmc_metadata["system"])
# sa = nk.sampler.MetropolisLocal(hi, n_chains=64)
# vs = nk.vqs.MCState(sa, build_model(vmc_metadata, g), n_samples=2**10, n_discard_per_chain=16)
# gs = nk.driver.VMC_SR(ha, nk.optimizer.Sgd(learning_rate=7.5e-4), variational_state=vs, diag_shift=5e-4, momentum=0, use_ntk=False)
# log = nk.logging.RuntimeLog()
# gs.run(n_iter=vmc_metadata["vmc"]["n_iters"], out=log, obs=obs, callback=log_parameters)
#
# VMC_dir = f"Results/MixedIsing_L12_hx1.25_hz0.15_RBMSymm_alpha2_{datetime.now().strftime('%m%d_%H%M')}"
# os.makedirs(VMC_dir, exist_ok=True)
# log.serialize(f"{VMC_dir}/VMC_log.json")
# with open(f"{VMC_dir}/VMC_metadata.json", "w") as f:
#     json.dump(vmc_metadata, f, indent=4)

with open(f"{VMC_dir}/VMC_metadata.json") as f:
    vmc_metadata = json.load(f)
total_iters = vmc_metadata["vmc"]["n_iters"]

# VMC iterations to time evolve (one after the other)
N_states = 20
iter_list = [int((i + 1) * total_iters // N_states - 1) for i in range(N_states)]
# iter_list = [149, 1499, 2999]


# ============================================================
# TDVP settings
# ============================================================
dt = 0.01
tf = 10.0
solver = nkx.dynamics.RK4(dt=dt)
# solver = nkx.dynamics.Euler(dt=dt)
n_samples = 2**11

# Sampler used during the time evolution
n_chains = 64
n_discard_per_chain = 16

# Observables logged at every time step
obs_names = ["Energy", "corr_x_avg", "avg_mag_x"]

# Log every dt after t = 0 (same times as p-tVMC)
tstops = np.linspace(dt, tf, int(round(tf / dt)))


# ============================================================
# Rebuild system and model
# ============================================================
g, hi, ha, obs = build_system(vmc_metadata["system"])
obs = {name: obs[name] for name in obs_names}
ma = build_model(vmc_metadata, g)

sa = nk.sampler.MetropolisHamiltonian(hi, ha, n_chains=n_chains)
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

# Load VMC parameter history
log = nk.utils.history.HistoryDict.from_file(f"{VMC_dir}/VMC_log.json")
pars_log = log["Parameters"]

if iter_list[0] < 0 or iter_list[-1] >= total_iters:
    raise ValueError(f"iter_list must be in [0, {total_iters})")


# ============================================================
# Output directory and metadata
# ============================================================
date = datetime.now().strftime("%Y%m%d_%H%M%S")
TDVP_directory = f"{VMC_dir}/TDVP_{solver.tableau}_tf{tf}_dt{dt}_ns{n_samples}_{date}"

tdvp_metadata = {
    "type": "TDVPSchmitt",
    "solver": str(solver.tableau),
    "error_norm": "qgt",
    "tf": tf,
    "dt": dt,
    "n_steps": int(tf / dt),
    "n_samples": n_samples,
    "sampler": {
        "type": "MetropolisHamiltonian",
        "n_chains": n_chains,
        "n_discard_per_chain": n_discard_per_chain,
    },
    "iter_list": iter_list,
    "total_iters": total_iters,
    "VMC_dir": VMC_dir,
    "FORWARD": FORWARD,
    "BACKWARD": BACKWARD,
    "observables": obs_names,
}

if SAVING:
    os.makedirs(TDVP_directory, exist_ok=True)
    with open(f"{TDVP_directory}/TE_metadata.json", "w") as f:
        json.dump(tdvp_metadata, f, indent=4)


# ============================================================
# Time evolve every state in iter_list (forward, then backward)
# ============================================================
t_start = time.time()
for ITER in iter_list:
    start_pars = params_at_iter(pars_log, ITER)
    run_dir = f"{TDVP_directory}/iter_{ITER}"
    if SAVING:
        os.makedirs(run_dir, exist_ok=True)
    print(f"Starting ITER={ITER}", flush=True)
    t0 = time.time()

    if FORWARD:
        vs.parameters = start_pars

        te_forward = nkx.driver.TDVPSchmitt(
            ha,
            variational_state=vs,
            ode_solver=solver,
            t0=0.0,
            error_norm="qgt",
        )

        log_forward = nk.logging.RuntimeLog()
        te_forward.run(
            T=tf,
            out=log_forward,
            obs=obs,
            tstops=tstops,
            callback=log_parameters,
        )

        if SAVING:
            log_forward.serialize(f"{run_dir}/log_forward_{ITER}-{total_iters}.json")

    if BACKWARD:
        # Evolving with -H forward in time is the backward time evolution
        vs.parameters = start_pars

        te_backward = nkx.driver.TDVPSchmitt(
            -1 * ha,
            variational_state=vs,
            ode_solver=solver,
            t0=0.0,
            error_norm="qgt",
        )

        log_backward = nk.logging.RuntimeLog()
        te_backward.run(
            T=tf,
            out=log_backward,
            obs=obs,
            tstops=tstops,
            callback=log_parameters,
        )

        if SAVING:
            log_backward.serialize(f"{run_dir}/log_backward_{ITER}-{total_iters}.json")

    print(f"Finished ITER={ITER} in {time.time() - t0:.2f} s", flush=True)

tdvp_metadata["runtime_seconds"] = time.time() - t_start

if SAVING:
    with open(f"{TDVP_directory}/TE_metadata.json", "w") as f:
        json.dump(tdvp_metadata, f, indent=4)
    print(f"Saved TDVP runs under: {TDVP_directory}")
