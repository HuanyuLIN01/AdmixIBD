"""AdmixIBD: population admixture inference from IBD segment lengths.

This refactor preserves the original truncated formulas, BOBYQA settings,
three optimization stages, and legacy JSON output field names.
"""

import argparse
import json
from math import exp
from pathlib import Path

import nlopt
import numpy as np


PARAMETERS = ("N_hap", "tau_a", "m")
THRESHOLD_MORGAN = 0.01  # 1 cM
RECOMBINATION_RATE = 1.0  # Original model convention


def load_ibd_lengths(path, replicates):
    """Read cM segment lengths and convert to Morgan.

    For compatibility, each entry in 'rep' repeats the same input file,
    matching the original script's behavior (rep values were unused).
    """
    with open(path, encoding="utf-8") as handle:
        lengths = [float(line) * 1e-2 for line in handle if line.strip()]
    return np.asarray(lengths * len(replicates), dtype=float)


def get_parameters(x, config):
    """Insert optimized values into the fixed/free parameter positions."""
    if tuple(config["fix_name"]) != PARAMETERS:
        raise ValueError(f"fix_name must be ordered as {PARAMETERS}")
    values = iter(x)
    return {
        name: (config["fix_val"][i] if config["fix"][i] == 1 else next(values))
        for i, name in enumerate(PARAMETERS)
    }


def evaluate(x, data, config):
    """Return total loss, unweighted NLL, sharing loss, and IBD sharing."""
    pars = get_parameters(x, config)
    n_hap, tau, m = (pars[name] for name in PARAMETERS)
    u, r = THRESHOLD_MORGAN, RECOMBINATION_RATE

    # Original truncated IBD-length density (do not change its formula).
    density = []
    for s in data:
        a = 2 * u**2 * (1 - 2 * m * (1 - m) *
                       (2 * (r * s * tau)**2 + 2 * r * s * tau + 1) *
                       exp(-2 * r * s * tau))
        b = s**3 * (1 - 2 * m * (1 - m) *
                    (1 + 2 * r * u * tau) * exp(-2 * r * u * tau))
        density.append(a / b)

    # Original truncated IBD-sharing formula.
    ibd = (1.0 / n_hap) * (
        1.0 / (r * u)
        - 2 * m * (1 - m) * (1 + r * u * tau) / (r * u)
        * exp(-2 * r * u * tau)
    )
    nll = -np.sum(np.log(density))
    sharing_loss = (ibd - config["ibd_label"]) ** 2
    total_loss = config["beta"] * nll + config["gamma"] * sharing_loss
    return total_loss, nll, sharing_loss, ibd


def optimize_stage(data, config):
    """Run the original BOBYQA optimization with fixed/free parameters."""
    free_indices = [i for i, fixed in enumerate(config["fix"]) if fixed == 0]
    if not free_indices:
        raise ValueError("At least one parameter must remain free")

    optimizer = nlopt.opt(nlopt.LN_BOBYQA, len(free_indices))
    optimizer.set_min_objective(lambda x, grad: evaluate(x, data, config)[0])
    optimizer.set_xtol_rel(1e-5)
    optimizer.set_lower_bounds([config[f"{PARAMETERS[i]}_bound"][0]
                                for i in free_indices])
    optimizer.set_upper_bounds([config[f"{PARAMETERS[i]}_bound"][1]
                                for i in free_indices])
    initial = [config["init_par"][i] for i in free_indices]
    optimum = optimizer.optimize(initial)
    min_loss = optimizer.last_optimum_value()
    _, nll, sharing_loss, ibd = evaluate(optimum, data, config)
    parameters = get_parameters(optimum, config)

    # Retain the published sample JSON keys, including original spelling.
    return {
        **config,
        "opt N_hap aproximated": parameters["N_hap"],
        "opt tau_a aproximated": parameters["tau_a"],
        "opt m aproximated": parameters["m"],
        "opt loss": min_loss,
        "loss1": nll,
        "loss2": sharing_loss,
        "ibd aproximated": ibd,
    }



def run(config, data, max_iter=30, tol=1e-5):
    """Alternating optimization of demographic parameters."""

    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if tol <= 0:
        raise ValueError("tol must be positive")

    original_fix = config["fix"]

    # Initialize parameters, respecting fixed values.
    current = {
        name: float(
            config["fix_val"][i]
            if original_fix[i] == 1
            else config["init_par"][i]
        )
        for i, name in enumerate(PARAMETERS)
    }

    def summarize():
        """Evaluate the original joint objective."""
        free_x = [
            current[name]
            for i, name in enumerate(PARAMETERS)
            if original_fix[i] == 0
        ]

        loss, nll, sharing_loss, ibd = evaluate(
            free_x, data, config
        )

        return {
            **{
                f"opt {name} aproximated": current[name]
                for name in PARAMETERS
            },
            "opt loss": float(loss),
            "loss1": float(nll),
            "loss2": float(sharing_loss),
            "ibd aproximated": float(ibd),
        }

    # Round 0 records initial parameters before optimization.
    round0 = {**config, **summarize()}

    history = []
    converged = False
    last_tau_stage = None
    last_nm_stage = None

    for iteration in range(1, max_iter + 1):
        previous = current.copy()

        # Step 1: Update admixture time (tau_a).
        # Keep N_hap and m fixed.
        if original_fix[1] == 0:
            tau_config = {
                **config,
                "fix": [1, 0, 1],
                "init_par": [
                    current[p] for p in PARAMETERS
                ],
                "fix_val": [
                    current[p] for p in PARAMETERS
                ],
            }

            last_tau_stage = optimize_stage(
                data, tau_config
            )

            current["tau_a"] = float(
                last_tau_stage["opt tau_a aproximated"]
            )

        # Step 2: Update N_hap and/or m.
        # Respect the original fixed-parameter settings.
        nm_fix = [
            original_fix[0],
            1,
            original_fix[2],
        ]

        if 0 in (nm_fix[0], nm_fix[2]):
            nm_config = {
                **config,
                "fix": nm_fix,
                "init_par": [
                    current[p] for p in PARAMETERS
                ],
                "fix_val": [
                    current[p] for p in PARAMETERS
                ],
            }

            last_nm_stage = optimize_stage(
                data, nm_config
            )

            for i in (0, 2):
                if original_fix[i] == 0:
                    name = PARAMETERS[i]
                    current[name] = float(
                        last_nm_stage[
                            f"opt {name} aproximated"
                        ]
                    )

        # Check convergence using free parameters only.
        max_change = max(
            abs(current[name] - previous[name])
            / max(1.0, abs(previous[name]))
            for i, name in enumerate(PARAMETERS)
            if original_fix[i] == 0
        )

        summary = summarize()

        history.append({
            "iteration": iteration,
            **summary,
            "max_relative_change": float(max_change),
        })

        print(
            f"Iteration {iteration:02d}: "
            f"N_hap={current['N_hap']:.4f}, "
            f"tau_a={current['tau_a']:.4f}, "
            f"m={current['m']:.6f}, "
            f"loss={summary['opt loss']:.8f}, "
            f"change={max_change:.2e}"
        )

        if max_change < tol:
            converged = True
            print(f"Converged after {iteration} iterations.")
            break

    if not converged:
        print(
            f"Maximum iterations ({max_iter}) reached "
            "without satisfying the convergence tolerance."
        )

    return {
        "round0": round0,
        "round t1": last_tau_stage,
        "round mn1": last_nm_stage,
        "iterations": history,
        "final": summarize(),
        "n_iterations": len(history),
        "converged": converged,
    }


def main():
    parser = argparse.ArgumentParser(description="AdmixIBD inference")
    parser.add_argument("--par-path", default="examples/input_par.json",
                        help="Path to the input JSON configuration")
    parser.add_argument("--result-path", default="examples/result.json",
                        help="Where to save the result JSON")
    parser.add_argument("--max-iter", type=int, default=30,
                        help="Maximum number of alternating iterations")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Convergence tolerance")
    args = parser.parse_args()

    with open(args.par_path, encoding="utf-8") as handle:
        config = json.load(handle)
    data = load_ibd_lengths(config["data_path"], config["rep"])
    result = run(
        config,
        data,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
