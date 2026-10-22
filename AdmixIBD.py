"""AdmixIBD: reference-free population admixture inference from IBD segments.

Alternating optimization using the original truncated IBD-length density,
IBD-sharing formula, and NLopt BOBYQA optimizer.

Outputs two files under --result-path:
  <prefix>.json : complete optimization record
  <prefix>.txt  : final estimates only
"""

import argparse
import json
from pathlib import Path

import nlopt
import numpy as np


PARAMETERS = ("N_hap", "tau_a", "m")
THRESHOLD_MORGAN = 0.01  # 1 cM
RECOMBINATION_RATE = 1.0  # Original model convention
INVALID_LOSS = 1e100


def load_ibd_lengths(path, replicates):
    """Read one segment length per line (cM) and convert to Morgans.

    For compatibility with the original implementation, each item in 'rep'
    repeats the same file; the values themselves are not used.
    """
    with open(path, encoding="utf-8") as handle:
        lengths = [float(line) * 0.01 for line in handle if line.strip()]

    if not lengths:
        raise ValueError(f"No IBD segment lengths found in {path}")
    if not np.all(np.isfinite(lengths)) or any(length <= 0 for length in lengths):
        raise ValueError("All IBD segment lengths must be positive and finite")
    if not isinstance(replicates, list) or not replicates:
        raise ValueError("'rep' must be a non-empty list (e.g. [1])")

    return np.asarray(lengths * len(replicates), dtype=float)


def validate_config(config):
    """Check fields needed for estimation without changing model settings."""
    if tuple(config.get("fix_name", PARAMETERS)) != PARAMETERS:
        raise ValueError(f"fix_name must have the order {PARAMETERS}")

    for key in ("fix", "fix_val", "init_par"):
        if len(config[key]) != len(PARAMETERS):
            raise ValueError(f"'{key}' must have exactly three entries")
    if any(flag not in (0, 1) for flag in config["fix"]):
        raise ValueError("Each 'fix' entry must be 0 (free) or 1 (fixed)")
    if all(config["fix"]):
        raise ValueError("At least one parameter must be free")

    for i, name in enumerate(PARAMETERS):
        lower, upper = config[f"{name}_bound"]
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Invalid bounds for {name}")
        value = config["fix_val"][i] if config["fix"][i] else config["init_par"][i]
        if value is None or not np.isfinite(value) or not (lower <= value <= upper):
            raise ValueError(f"Invalid initial/fixed value for {name}: {value}")

    if config["gamma"] < 0 or config["beta"] < 0:
        raise ValueError("gamma and beta must be non-negative")
    if config["gamma"] == 0 and config["beta"] == 0:
        raise ValueError("At least one of gamma and beta must be positive")
    if not np.isfinite(config["ibd_label"]):
        raise ValueError("ibd_label must be finite")


def get_parameters(x, config):
    """Place free optimizer variables and fixed values in parameter order."""
    values = iter(x)
    return {
        name: (config["fix_val"][i] if config["fix"][i] == 1 else next(values))
        for i, name in enumerate(PARAMETERS)
    }


def evaluate(x, data, config):
    """Return total loss, unweighted NLL, sharing loss, and predicted sharing."""
    pars = get_parameters(x, config)
    n_hap, tau, m = (pars[name] for name in PARAMETERS)
    u, r = THRESHOLD_MORGAN, RECOMBINATION_RATE

    # Avoid division by zero at the allowed lower bound N_hap=0.
    if n_hap <= 0 or tau < 0 or not (0 <= m <= 1):
        return INVALID_LOSS, INVALID_LOSS, INVALID_LOSS, float("nan")

    # Original truncated IBD-length density; vectorized over lengths.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a = 2 * u**2 * (
            1 - 2 * m * (1 - m)
            * (2 * (r * data * tau)**2 + 2 * r * data * tau + 1)
            * np.exp(-2 * r * data * tau)
        )
        b = data**3 * (
            1 - 2 * m * (1 - m)
            * (1 + 2 * r * u * tau) * np.exp(-2 * r * u * tau)
        )
        density = a / b

        # Original truncated IBD-sharing formula.
        ibd = (1.0 / n_hap) * (
            1.0 / (r * u)
            - 2 * m * (1 - m) * (1 + r * u * tau) / (r * u)
            * np.exp(-2 * r * u * tau)
        )

        if not np.all(np.isfinite(density)) or np.any(density <= 0):
            return INVALID_LOSS, INVALID_LOSS, INVALID_LOSS, float("nan")
        nll = -float(np.sum(np.log(density)))
        sharing_loss = float((ibd - config["ibd_label"])**2)
        total_loss = float(config["beta"] * nll + config["gamma"] * sharing_loss)

    if not np.isfinite(total_loss):
        return INVALID_LOSS, INVALID_LOSS, INVALID_LOSS, float("nan")
    return total_loss, nll, sharing_loss, float(ibd)


def optimize_stage(data, config):
    """Optimize currently free parameters using NLopt BOBYQA."""
    free_indices = [i for i, fixed in enumerate(config["fix"]) if fixed == 0]
    if not free_indices:
        raise ValueError("At least one parameter must be free in an optimization stage")

    optimizer = nlopt.opt(nlopt.LN_BOBYQA, len(free_indices))
    optimizer.set_min_objective(lambda x, grad: evaluate(x, data, config)[0])
    optimizer.set_xtol_rel(1e-5)
    optimizer.set_lower_bounds([
        config[f"{PARAMETERS[i]}_bound"][0] for i in free_indices
    ])
    optimizer.set_upper_bounds([
        config[f"{PARAMETERS[i]}_bound"][1] for i in free_indices
    ])
    initial = [config["init_par"][i] for i in free_indices]
    optimum = optimizer.optimize(initial)
    loss, nll, sharing_loss, ibd = evaluate(optimum, data, config)
    if loss >= INVALID_LOSS:
        raise RuntimeError("Optimization ended at an invalid parameter combination")
    pars = get_parameters(optimum, config)

    # Retain the original output key spelling for compatibility.
    return {
        **config,
        **{f"opt {name} aproximated": float(pars[name]) for name in PARAMETERS},
        "opt loss": float(loss),
        "loss1": float(nll),
        "loss2": float(sharing_loss),
        "ibd aproximated": float(ibd),
    }


def run(config, data, max_iter=30, tol=1e-5):
    """Alternate updates of tau_a and (N_hap, m) under one joint loss."""
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter must be >= 1 and tol must be positive")
    validate_config(config)
    original_fix = list(config["fix"])
    current = {
        name: float(
            config["fix_val"][i] if original_fix[i] else config["init_par"][i]
        )
        for i, name in enumerate(PARAMETERS)
    }

    def summarize():
        free_x = [
            current[name] for i, name in enumerate(PARAMETERS)
            if original_fix[i] == 0
        ]
        loss, nll, sharing_loss, ibd = evaluate(free_x, data, config)
        if loss >= INVALID_LOSS:
            raise RuntimeError("Current parameter estimates are invalid")
        return {
            **{f"opt {name} aproximated": current[name] for name in PARAMETERS},
            "opt loss": loss,
            "loss1": nll,
            "loss2": sharing_loss,
            "ibd aproximated": ibd,
        }

    round0 = {**config, **summarize()}
    history = []
    last_tau_stage = None
    last_nm_stage = None
    converged = False

    for iteration in range(1, max_iter + 1):
        previous = current.copy()

        # 1. Update admixture time, keeping N_hap and m fixed.
        if original_fix[1] == 0:
            tau_config = {
                **config,
                "fix": [1, 0, 1],
                "init_par": [current[name] for name in PARAMETERS],
                "fix_val": [current[name] for name in PARAMETERS],
            }
            last_tau_stage = optimize_stage(data, tau_config)
            current["tau_a"] = last_tau_stage["opt tau_a aproximated"]
        after_tau = summarize()

        # 2. Update the originally free N_hap and/or m, keeping tau_a fixed.
        nm_fix = [original_fix[0], 1, original_fix[2]]
        if nm_fix[0] == 0 or nm_fix[2] == 0:
            nm_config = {
                **config,
                "fix": nm_fix,
                "init_par": [current[name] for name in PARAMETERS],
                "fix_val": [current[name] for name in PARAMETERS],
            }
            last_nm_stage = optimize_stage(data, nm_config)
            for i in (0, 2):
                if original_fix[i] == 0:
                    name = PARAMETERS[i]
                    current[name] = last_nm_stage[f"opt {name} aproximated"]
        after_nm = summarize()

        max_change = max(
            abs(current[name] - previous[name]) / max(1.0, abs(previous[name]))
            for i, name in enumerate(PARAMETERS) if original_fix[i] == 0
        )
        history.append({
            "iteration": iteration,
            "after_tau_update": after_tau,
            "after_N_m_update": after_nm,
            "max_relative_change": float(max_change),
        })
        print(
            f"Iteration {iteration:02d}: "
            f"N_hap={current['N_hap']:.4f}, "
            f"tau_a={current['tau_a']:.4f}, "
            f"m={current['m']:.6f}, "
            f"loss={after_nm['opt loss']:.8f}, "
            f"change={max_change:.2e}"
        )
        if max_change < tol:
            converged = True
            break

    return {
        "round0": round0,
        "round t1": last_tau_stage,
        "round mn1": last_nm_stage,
        "iterations": history,
        "final": summarize(),
        "n_iterations": len(history),
        "converged": converged,
    }


def format_final_result(result):
    """Format final estimates for terminal and plain-text output."""
    final = result["final"]
    return "\n".join([
        "AdmixIBD Final Results",
        "======================",
        f"N_hap             : {final['opt N_hap aproximated']:.6f}",
        f"tau_a (generations): {final['opt tau_a aproximated']:.6f}",
        f"m                 : {final['opt m aproximated']:.6f}",
        f"IBD sharing       : {final['ibd aproximated']:.10f}",
        f"Loss              : {final['opt loss']:.8f}",
        f"Iterations        : {result['n_iterations']}",
        f"Converged         : {result['converged']}",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description="AdmixIBD inference")
    parser.add_argument(
        "--par-path", default="examples/input_par.json",
        help="Path to input JSON configuration",
    )
    parser.add_argument(
        "--result-path", default=".",
        help="Output directory for the JSON and TXT files",
    )
    parser.add_argument(
        "--prefix", default="result",
        help="Common file prefix (creates <prefix>.json and <prefix>.txt)",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    if not args.prefix or args.prefix in (".", "..") or Path(args.prefix).name != args.prefix:
        parser.error("--prefix must be a file name, not a path")
    output_dir = Path(args.result_path)
    if output_dir.exists() and not output_dir.is_dir():
        parser.error("--result-path must be a directory, not a file")

    with open(args.par_path, encoding="utf-8") as handle:
        config = json.load(handle)
    data = load_ibd_lengths(config["data_path"], config["rep"])
    result = run(config, data, max_iter=args.max_iter, tol=args.tol)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.prefix}.json"
    txt_path = output_dir / f"{args.prefix}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4, allow_nan=False)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(format_final_result(result))

    print("\n" + format_final_result(result), end="")
    print(f"Full results:  {json_path}")
    print(f"Final summary: {txt_path}")


if __name__ == "__main__":
    main()
