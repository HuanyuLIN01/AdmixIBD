# AdmixIBD

**Reference-free inference of population admixture history from identity-by-descent (IBD) segments**

![AdmixIBD conceptual illustration](docs/AdmixIBD_schematic.png)

## Overview

AdmixIBD is a reference-free statistical framework for inferring population admixture history from identity-by-descent (IBD) segments. Based on the coalescent process under the Hybrid Isolation (HI) model, AdmixIBD exploits the effects of population divergence and admixture on genome-wide IBD patterns to infer demographic parameters without requiring predefined ancestral reference populations.

The framework integrates two complementary IBD summaries: the **IBD-sharing ratio**, which provides information about effective population size and admixture proportion, and the **conditional length distribution of detectable IBD segments**, which is particularly informative about admixture time. AdmixIBD uses alternating optimization to update admixture time and the remaining free demographic parameters from IBD segments detected within the target admixed population.

## Installation

AdmixIBD requires Python 3 and the packages `numpy`, `scipy`, and `nlopt`.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/HuanyuLIN01/AdmixIBD.git
cd AdmixIBD
python -m pip install -r requirements.txt
```

## Quick Start

The `examples/` directory contains sample IBD segment lengths and a configuration file. From the repository root, run:

```bash
python AdmixIBD.py \
    --par-path examples/input_par.json \
    --prefix example
```

Both **`--par-path`** (input configuration file) and **`--prefix`** (output filename prefix) are required. By default, AdmixIBD writes output files to the **current working directory**:

```text
example.json
example.txt
```

To use a different output directory, specify `--result-path`:

```bash
python AdmixIBD.py \
    --par-path examples/input_par.json \
    --result-path examples \
    --prefix example
```

This creates `examples/example.json` and `examples/example.txt`.

Additional optional arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--result-path` | `.` | Output directory for both files |
| `--max-iter` | `30` | Maximum number of alternating optimization iterations |
| `--tol` | `1e-5` | Relative parameter-change convergence tolerance |

## Input

AdmixIBD requires **IBD segment lengths** and a **JSON model configuration file**.

### 1. IBD Segment Lengths

The file `examples/input.IBDlength_cM.txt` provides one IBD segment length per line, measured in centimorgans (cM). The script converts these lengths to Morgans internally.

### 2. Model Configuration

The JSON file specified using `--par-path` contains the input file path and model settings. In most applications, users should update **`data_path`** and **`ibd_label`**. When fixing a demographic parameter, also update **`fix`** and **`fix_val`**.

| Parameter | Description | Value in the supplied example |
| --- | --- | --- |
| `data_path` | Path to the IBD segment-length file | `examples/input.IBDlength_cM.txt` |
| `ibd_label` | Observed IBD-sharing ratio | `0.0035686933476118` |
| `N_hap_bound` | Bounds for the effective population size parameter | `[0, 100000]` |
| `tau_a_bound` | Bounds for admixture time (generations) | `[0, 500]` |
| `m_bound` | Bounds for admixture proportion | `[0, 0.5]` |
| `init_par` | Initial values in the order `N_hap`, `tau_a`, `m` | `[10000, 100, 0.25]` |
| `fix` | Parameter flags: `0` = estimated; `1` = fixed | `[0, 0, 1]` |
| `fix_val` | Values for fixed parameters (`null` for free parameters) | `[null, null, 0.480411840425532]` |
| `fix_name` | Parameter names and order | `["N_hap", "tau_a", "m"]` |
| `gamma` | Weight on the IBD-sharing term | `10000` |
| `beta` | Weight on the IBD length-distribution term | `0.001` |
| `rep` | Compatibility setting; `[1]` for one copy of the input file | `[1]` |

The example **fixes the admixture proportion** (`m = 0.480411840425532`) while estimating `N_hap` and `tau_a`:

```json
"fix": [0, 0, 1],
"fix_val": [null, null, 0.480411840425532]
```

To estimate all three parameters instead, use:

```json
"fix": [0, 0, 0],
"fix_val": [null, null, null]
```

An independently estimated admixture proportion, for example from an analysis using [ADMIXTURE](https://dalexander.github.io/admixture/), may be supplied as a fixed value when its interpretation is compatible with the HI model. The provided numerical values are **example settings**, not general-purpose defaults.

**Important:** `data_path` is resolved relative to the directory from which you run the command, **not** relative to the configuration JSON file. The quick-start examples assume commands are run from the repository root.

The fields `N_hap_true`, `tau_a_true`, and `m_true` are not required; simulation truth should not be supplied for real-data inference.

## Output

Each run creates **two files** using the required `--prefix` in the directory selected by `--result-path` (default: `.`):

- **`<prefix>.json`** — Full optimization record, including initial estimates, successive updates, iteration history, objective values, final estimates, and convergence status.
- **`<prefix>.txt`** — A concise summary of **final parameter estimates**, predicted IBD-sharing ratio, loss, number of iterations, and convergence status.

The final summary is also printed to the terminal. The TXT file is the easiest place to read the final results; the JSON file is intended for detailed inspection and reproducibility.

Some JSON field names retain the legacy spelling `aproximated` for compatibility with earlier output.

## Application to the 1000 Genomes Project

The `KGP/` directory contains supplementary inputs and results from the real-data applications described in the manuscript, including IBD segment lengths, IBD-sharing ratios, admixture proportions, and associated inference results.

## Citation

A manuscript describing AdmixIBD is in preparation. Citation details will be added when available.
