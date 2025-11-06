"""
microsim_sirs.py

A simple individual-based (microsimulation) SIRS model.

States:
 - 0 : Susceptible (S)
 - 1 : Infected    (I)
 - 2 : Recovered   (R)

At each discrete time step:
 - Each susceptible has a probability of getting infected that depends
   on the number of current infected individuals, the per-contact
   transmission probability (beta), and the average number of contacts
   per individual per time step (contacts).
   p_infect = 1 - (1 - beta)^(contacts * I / N)
 - Each infected recovers with probability gamma per time step.
 - Each recovered loses immunity (becomes susceptible) with probability delta per time step.

Functions:
 - run_simulation(...) -> runs a single microsimulation, returns time series of S, I, R counts
 - run_ensemble(...) -> runs multiple realizations and returns arrays (n_runs x (T+1)) of S,I,R
 - plot_timeseries(...) -> optional plotting helper (requires matplotlib)

Example:
    python microsim_sirs.py
    # or import functions and call run_simulation(...)
"""

from typing import Tuple
import numpy as np


def run_simulation(
    N,
    beta,
    contacts,
    gamma,
    delta,
    steps,
    I0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single stochastic (microsimulation) SIRS model.

    Parameters
    ----------
    N : int
        Total number of individuals (fixed).
    beta : float
        Per-contact transmission probability (0 <= beta <= 1).
    contacts : float
        Mean number of contacts per individual per time step.
    gamma : float
        Per-time-step recovery probability for infected individuals (0 <= gamma <= 1).
    delta : float
        Per-time-step loss-of-immunity probability for recovered individuals (0 <= delta <= 1).
    steps : int
        Number of discrete time steps to simulate.
    I0 : int
        Initial number of infected individuals (randomly chosen).
    Note
    ----
    This function does not accept an RNG seed; a non-deterministic RNG is used.

    Returns
    -------
    S_ts, I_ts, R_ts : (np.ndarray, np.ndarray, np.ndarray)
        Arrays of length steps+1 with the counts of Susceptible, Infected, Recovered at each time step.
    """
    # Use a non-deterministic RNG (no explicit seed)
    rng = np.random.default_rng()

    # States array for each individual: 0=S, 1=I, 2=R
    states = np.zeros(N, dtype=np.int8)
    # seed initial infections
    if I0 > 0:
        init_infected = rng.choice(np.arange(N), size=min(I0, N), replace=False)
        states[init_infected] = 1

    S_ts = np.empty(steps + 1, dtype=int)
    I_ts = np.empty(steps + 1, dtype=int)
    R_ts = np.empty(steps + 1, dtype=int)

    def counts():
        # Count S, I, R
        unique, cnts = np.unique(states, return_counts=True)
        # Map counts to S,I,R
        s = cnts[unique.tolist().index(0)] if 0 in unique else 0
        i = cnts[unique.tolist().index(1)] if 1 in unique else 0
        r = cnts[unique.tolist().index(2)] if 2 in unique else 0
        return s, i, r

    S_ts[0], I_ts[0], R_ts[0] = counts()

    for t in range(1, steps + 1):
        s, i, r = S_ts[t - 1], I_ts[t - 1], R_ts[t - 1]

        # 1) Infection step
        if i == 0 or s == 0 or beta <= 0 or contacts <= 0:
            new_infections = 0
        else:
            # probability a susceptible avoids infection from a single contact is (1 - beta).
            # Approximate number of infectious contacts per susceptible: contacts * (i / N)
            # so probability to avoid infection over all contacts: (1 - beta)^(contacts * i / N)
            # hence p_infect same for all susceptibles:
            lam = contacts * (i / N)
            # numerical safety
            if lam <= 0:
                p_infect = 0.0
            else:
                # Use log/expm1 to manage numerical issues when beta small/large lam
                p_infect = 1.0 - (1.0 - beta) ** lam
                # Ensure p_infect within [0,1]
                p_infect = min(max(p_infect, 0.0), 1.0)

            # Number of new infections among susceptibles is Binomial(s, p_infect)
            new_infections = int(rng.binomial(s, p_infect))

        # Infect particular individuals (choose randomly among susceptibles)
        if new_infections > 0:
            sus_idx = np.nonzero(states == 0)[0]
            chosen = rng.choice(sus_idx, size=new_infections, replace=False)
            states[chosen] = 1

        # 2) Recovery step: each infected recovers with probability gamma
        if i > 0 and gamma > 0:
            infected_idx = np.nonzero(states == 1)[0]
            num_recover = int(rng.binomial(len(infected_idx), gamma))
            if num_recover > 0:
                chosen_rec = rng.choice(infected_idx, size=num_recover, replace=False)
                states[chosen_rec] = 2

        # 3) Waning immunity: recovered -> susceptible with probability delta
        if r > 0 and delta > 0:
            rec_idx = np.nonzero(states == 2)[0]
            num_wane = int(rng.binomial(len(rec_idx), delta))
            if num_wane > 0:
                chosen_wane = rng.choice(rec_idx, size=num_wane, replace=False)
                states[chosen_wane] = 0

        S_ts[t], I_ts[t], R_ts[t] = counts()

    return S_ts, I_ts, R_ts


def run_ensemble(
    n_runs: int = 50,
    **sim_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run an ensemble of stochastic realizations.

    Parameters
    ----------
    n_runs : int
        Number of independent realizations.
    sim_kwargs : dict
        Arguments passed to run_simulation (except rng_seed).
        If you want reproducible ensemble, provide rng_seed as an integer; each run will be seeded deterministically.

    Returns
    -------
    S_ens, I_ens, R_ens : np.ndarray
        Arrays of shape (n_runs, steps+1) containing counts for each realization.
    """
    # run one simulation to get length
    tmp = run_simulation(**sim_kwargs)
    steps_plus_one = tmp[0].size

    S_ens = np.empty((n_runs, steps_plus_one), dtype=int)
    I_ens = np.empty((n_runs, steps_plus_one), dtype=int)
    R_ens = np.empty((n_runs, steps_plus_one), dtype=int)

    for run in range(n_runs):
        S_ts, I_ts, R_ts = run_simulation(**sim_kwargs)
        S_ens[run] = S_ts
        I_ens[run] = I_ts
        R_ens[run] = R_ts

    return S_ens, I_ens, R_ens


def plot_timeseries(
    S_ts: np.ndarray,
    I_ts: np.ndarray,
    R_ts: np.ndarray,
    title: str = "SIRS microsimulation",
    show: bool = True,
) -> None:
    """
    Plot S, I, R time series. Accepts either a single-run series (1D arrays) or ensemble (2D arrays).
    If input arrays are 2D, the function plots the mean and a light band for +/- one std.

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting (install with `pip install matplotlib`)")

    def _plot(ax, arr, label, color):
        if arr.ndim == 1:
            ax.plot(arr, label=label, color=color)
        elif arr.ndim == 2:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(mean, label=f"{label} (mean)", color=color)
            ax.fill_between(np.arange(mean.size), mean - std, mean + std, color=color, alpha=0.25)
        else:
            raise ValueError("Array must be 1D or 2D")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _plot(ax, S_ts, "Susceptible", "C0")
    _plot(ax, I_ts, "Infected", "C1")
    _plot(ax, R_ts, "Recovered", "C2")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Number of individuals")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if show:
        plt.show()


if __name__ == "__main__":
    # Minimal demo without argparse/CLI parsing.
    # Adjust these variables directly in the file or import the functions and call them from another script.
    N = 1000
    I0 = 10
    beta = 0.03
    contacts = 10.0
    gamma = 0.1
    delta = 0.01
    steps = 365*2
    runs = 5

    # Run an ensemble and attempt to plot; if matplotlib is not present, print a small summary instead.
    S_ens, I_ens, R_ens = run_ensemble(
        n_runs=runs,
        N=N,
        beta=beta,
        contacts=contacts,
        gamma=gamma,
        delta=delta,
        steps=steps,
        I0=I0,
    )

    try:
        plot_timeseries(S_ens, I_ens, R_ens, title=f"SIRS microsim (N={N}, runs={runs})", show=True)
    except RuntimeError:
        mean_I = I_ens.mean(axis=0)
        peak = mean_I.max()
        print(f"Peak mean infected: {peak:.1f}")
        print("Run complete. Install matplotlib to see plots (pip install matplotlib).")