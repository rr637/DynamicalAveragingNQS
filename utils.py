import os
import glob
import json

import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmaz

from models.Vit_systematic import ViT


# ============================================================
# Systems
# ------------------------------------------------------------
# To add a system: write a get_<system>_observables function that
# returns the Hamiltonian and a dictionary of observables, then add
# a branch for it in build_system
# ============================================================

def get_1D_Ising_observables(hi, L, hx, hz, J):

    # Transverse field term: -hx * sum_i σ^x_i
    ha = sum([-hx * sigmax(hi, i) for i in range(L)])

    # Longitudinal field term: -hz * sum_i σ^z_i
    # Breaks Z2 symmetry and biases spins along Z
    ha += sum([-hz * sigmaz(hi, i) for i in range(L)])

    # Ising interaction term: -J * sum_i σ^z_i σ^z_{i+1}
    # Periodic boundary ensures (L-1) couples to 0
    ha += sum([
        -J * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % L)
        for i in range(L)
    ])

    center = L // 2

    # Nearest-neighbour correlators at the center of the chain
    corr_x = sigmax(hi, center) @ sigmax(hi, center + 1)
    corr_z = sigmaz(hi, center) @ sigmaz(hi, center + 1)

    # Average magnetizations: m = (1/L) * sum_i <σ_i>
    avg_mag_x = (1 / L) * sum([sigmax(hi, i) for i in range(L)])
    avg_mag_z = (1 / L) * sum([sigmaz(hi, i) for i in range(L)])

    # Translation-averaged nearest-neighbour correlators: C_avg = (1/L) * sum_i σ_i σ_{i+1}
    # Same expectation value as corr_x, corr_z for a translation invariant state but lower MC variance
    corr_x_avg = (1 / L) * sum([sigmax(hi, i) @ sigmax(hi, (i + 1) % L) for i in range(L)])
    corr_z_avg = (1 / L) * sum([sigmaz(hi, i) @ sigmaz(hi, (i + 1) % L) for i in range(L)])

    obs = {
        "Energy": ha,
        "corr_x": corr_x,
        "corr_z": corr_z,
        "avg_mag_x": avg_mag_x,
        "avg_mag_z": avg_mag_z,
        "corr_x_avg": corr_x_avg,
        "corr_z_avg": corr_z_avg,
    }
    return ha, obs


def get_2D_Ising_observables(hi, g, hx, hz, J):
    N = g.n_nodes
    edges = list(g.edges())

    # H = -J sum_<ij> Z_i Z_j - hx sum_i X_i - hz sum_i Z_i
    ha = nk.operator.LocalOperator(hi, dtype=complex)
    for i, j in edges:
        ha += -J * (sigmaz(hi, i) @ sigmaz(hi, j))
    for i in range(N):
        ha += -hx * sigmax(hi, i)
        if hz != 0.0:
            ha += -hz * sigmaz(hi, i)

    # Average magnetizations and nearest-neighbour correlators averaged over all bonds
    avg_mag_x = sum(sigmax(hi, i) for i in range(N)) / N
    avg_mag_z = sum(sigmaz(hi, i) for i in range(N)) / N
    corr_zz = sum(sigmaz(hi, i) @ sigmaz(hi, j) for i, j in edges) / len(edges)
    corr_xx = sum(sigmax(hi, i) @ sigmax(hi, j) for i, j in edges) / len(edges)

    obs = {
        "Energy": ha,
        "avg_mag_x": avg_mag_x,
        "avg_mag_z": avg_mag_z,
        "corr_zz": corr_zz,
        "corr_xx": corr_xx,
    }
    return ha, obs


def build_system(system):
    # Build graph, Hilbert space, Hamiltonian and observables from metadata["system"]

    if system["model_name"] == "1D mixed-field Ising model":
        g = nk.graph.Hypercube(length=system["L"], n_dim=1, pbc=system["pbc"])
        hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
        ha, obs = get_1D_Ising_observables(hi, system["L"], system["hx"], system["hz"], system["J"])

    elif system["model_name"] == "2D mixed-field Ising model":
        g = nk.graph.Hypercube(length=system["L"], n_dim=2, pbc=system["pbc"])
        hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
        ha, obs = get_2D_Ising_observables(hi, g, system["hx"], system["hz"], system["J"])

    # elif system["model_name"] == "<your system>":
    #     g = ...
    #     hi = ...
    #     ha, obs = get_<your system>_observables(...)

    else:
        raise NotImplementedError(f"System {system['model_name']} not implemented")

    return g, hi, ha, obs


# ============================================================
# Models
# ------------------------------------------------------------
# To add a model: add a branch for it in build_model and save the
# hyperparameters it needs in metadata["model"] in VMC_Ising.py
# ============================================================

def build_model(metadata, g):
    # Build the neural network from metadata["model"]
    model = metadata["model"]

    if model["type"] == "RBMSymm":
        ma = nk.models.RBMSymm(
            symmetries=g.translation_group(),
            alpha=model["alpha"],
            param_dtype=complex if model["param_dtype"] == "complex" else float,
            use_visible_bias=model["use_visible_bias"],
            use_hidden_bias=model["use_hidden_bias"],
        )

    elif model["type"] == "ViT":
        ma = ViT(
            num_layers=model["num_layers"],
            d_model=model["d_model"],
            heads=model["n_heads"],
            b=model["patch_size"],
            Lx=metadata["system"]["L"],
            Ly=metadata["system"]["L"],
            expansion_factor=model["expansion_factor"],
            transl_invariant=model["transl_invariant"],
        )

    # elif model["type"] == "<your model>":
    #     ma = ...

    else:
        raise NotImplementedError(f"Model {model['type']} not implemented")

    return ma


# ============================================================
# Loading VMC and time evolution results
# ============================================================

def params_at_iter(pars_log, ITER):
    # Network parameters at iteration ITER of a log
    if "value" in pars_log:
        return pars_log["value"][ITER]
    return {key: params_at_iter(pars_log[key], ITER) for key in pars_log.keys()}


def get_TE_iters(TE_dir):
    # VMC iterations that were time evolved in TE_dir
    iters = [int(d.split("_")[-1]) for d in os.listdir(TE_dir) if d.startswith("iter_")]
    return sorted(iters)


def load_TE_log(TE_dir, ITER, direction):
    # Log and times of the forward or backward time evolution of VMC state ITER
    file = glob.glob(f"{TE_dir}/iter_{ITER}/log_{direction}_*.json")[0]
    log = nk.utils.history.HistoryDict.from_file(file)

    # p-tVMC logs the time as "times" and TDVP as "t"
    times = log["times"].value if "times" in log else log["t"].value
    return np.array(times), log


def load_trajectory(TE_dir, ITER, pars_log, dt_obs):
    # Parameters of the trajectory of VMC state ITER every dt_obs from -tf to tf
    # t = 0 is the VMC state, t < 0 the backward and t > 0 the forward time evolution
    with open(f"{TE_dir}/TE_metadata.json") as f:
        te_metadata = json.load(f)
    dt = te_metadata["dt"]
    stride = int(round(dt_obs / dt))

    times_f, log_f = load_TE_log(TE_dir, ITER, "forward")
    times_b, log_b = load_TE_log(TE_dir, ITER, "backward")
    idx_f = [i for i, t in enumerate(times_f) if int(round(t / dt)) % stride == 0]
    idx_b = [i for i, t in enumerate(times_b) if int(round(t / dt)) % stride == 0]

    times = np.concatenate((-times_b[idx_b][::-1], [0.0], times_f[idx_f]))
    params = (
        [params_at_iter(log_b["Parameters"], i) for i in idx_b[::-1]]
        + [params_at_iter(pars_log, ITER)]
        + [params_at_iter(log_f["Parameters"], i) for i in idx_f]
    )
    return times, params
