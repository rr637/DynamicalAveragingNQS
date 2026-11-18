import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["NETKET_SHARDING"] = "0"

import json
import time

import numpy as np
import jax.numpy as jnp
import netket as nk

from utils import build_system, build_model, params_at_iter, get_TE_iters, load_trajectory


# ============================================================
# User settings
# ============================================================
RESAMPLE_VMC = True     # resample VMC states
RESAMPLE_TE = True      # resample the time evolved states (p-tVMC or TDVP)

VMC_dir = "Results/MixedIsing_L12_hx1.25_hz0.15_RBMSymm_alpha2_0914_1641"
TE_dir = f"{VMC_dir}/PTVMC_PPE2()_tf10.0_dt0.01_ns2048_ni20_20260915_031633"

with open(f"{VMC_dir}/VMC_metadata.json") as f:
    metadata = json.load(f)
total_iters = metadata["vmc"]["n_iters"]

# VMC states: n_samples_VMC samples for each iteration in iter_list
N_states = 20
iter_list = [int((i + 1) * total_iters // N_states - 1) for i in range(N_states)]
n_samples_VMC = 2**22

# Time evolved states: n_samples_TE samples every dt_obs along the trajectory of every state in TE_iters
TE_iters = None         # None -> every state in TE_dir
n_samples_TE = 2**21
dt_obs = 0.1
TE_obs_names = ["corr_x_avg", "avg_mag_x"]
# Connected correlators <A> - <B>^2 whose errors are computed from the same samples
TE_conn_obs = {"conn_corr_x_avg": ("corr_x_avg", "avg_mag_x")}

n_chains = 4096
chunk_size = 2048


# ============================================================
# Rebuild system and model
# ============================================================
g, hi, ha, obs = build_system(metadata["system"])
ma = build_model(metadata, g)

log = nk.utils.history.HistoryDict.from_file(f"{VMC_dir}/VMC_log.json")
pars_log = log["Parameters"]


# ============================================================
# Resample VMC states
# ============================================================
if RESAMPLE_VMC:
    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples_VMC, n_discard_per_chain=32, chunk_size=chunk_size)

    # Same format as VMC_log.json at the resampled iterations
    log_resampled = nk.logging.RuntimeLog()
    t0 = time.time()
    for ITER in iter_list:
        vs.parameters = params_at_iter(pars_log, ITER)
        data = {name: vs.expect(op) for name, op in obs.items()}
        log_resampled(ITER, data)
        print(f"VMC iter {ITER}: Energy = {data['Energy']}  ({time.time() - t0:.0f} s)", flush=True)

    log_resampled.serialize(f"{VMC_dir}/VMC_log_resampled_ns{n_samples_VMC}.json")

    resampled_metadata = dict(metadata)
    resampled_metadata["resampled"] = {
        "n_samples": n_samples_VMC,
        "sampler": "MetropolisLocal",
        "n_chains": n_chains,
        "n_discard_per_chain": 32,
        "iter_list": iter_list,
    }
    with open(f"{VMC_dir}/VMC_metadata_resampled_ns{n_samples_VMC}.json", "w") as f:
        json.dump(resampled_metadata, f, indent=4)
    print(f"Saved {VMC_dir}/VMC_log_resampled_ns{n_samples_VMC}.json")


# ============================================================
# Resample time evolved states
# ============================================================
if RESAMPLE_TE:
    TE_obs = {name: obs[name] for name in TE_obs_names}

    if TE_iters is None:
        TE_iters = get_TE_iters(TE_dir)

    results = {}
    t0 = time.time()
    for ITER in TE_iters:
        times, params = load_trajectory(TE_dir, ITER, pars_log, dt_obs)

        sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
        vs = nk.vqs.MCState(sa, ma, n_samples=n_samples_TE, n_discard_per_chain=16, chunk_size=chunk_size)

        series = {name: np.zeros(len(times)) for name in list(TE_obs) + list(TE_conn_obs)}
        series_sigma = {name: np.zeros(len(times)) for name in list(TE_obs) + list(TE_conn_obs)}
        for i, pars in enumerate(params):
            # Fresh samples at every time step
            vs.parameters = pars
            vs.sample()

            # Local estimators with shape (n_chains, n_samples_per_chain)
            O_loc = {name: np.asarray(vs.local_estimators(op)).real for name, op in TE_obs.items()}
            for name in TE_obs:
                stats = nk.stats.statistics(jnp.asarray(O_loc[name]))
                series[name][i] = np.real(stats.mean)
                series_sigma[name][i] = np.real(stats.error_of_mean)

            # Error of <A> - <B>^2 from the local estimators of A - 2 <B> B
            for name, (A, B) in TE_conn_obs.items():
                stats = nk.stats.statistics(jnp.asarray(O_loc[A] - 2 * series[B][i] * O_loc[B]))
                series[name][i] = series[A][i] - series[B][i]**2
                series_sigma[name][i] = np.real(stats.error_of_mean)

        results[str(ITER)] = {
            "times": times.tolist(),
            "series": {name: series[name].tolist() for name in series},
            "series_sigma": {name: series_sigma[name].tolist() for name in series_sigma},
        }
        print(f"TE iter {ITER}: {len(times)} time steps  ({time.time() - t0:.0f} s)", flush=True)

    # Keep the states that were already resampled with the same settings
    TE_file = f"{TE_dir}/TE_log_resampled_dt{dt_obs}_ns{n_samples_TE}.json"
    if os.path.exists(TE_file):
        with open(TE_file) as f:
            results = {**json.load(f)["results"], **results}

    TE_resampled_metadata = {
        "TE_dir": TE_dir,
        "n_samples": n_samples_TE,
        "dt_obs": dt_obs,
        "sampler": "MetropolisLocal",
        "n_chains": n_chains,
        "n_discard_per_chain": 16,
        "observables": list(series.keys()),
        "connected_observables": TE_conn_obs,
    }
    with open(TE_file, "w") as f:
        json.dump({"metadata": TE_resampled_metadata, "results": results}, f, indent=1)
    print(f"Saved {TE_file}")
