# AdmixIBD

**Reference-free inference of population admixture history from identity-by-descent (IBD) segments**
![AdmixIBD conceptual illustration](docs/AdmixIBD_schematic.png)
AdmixIBD is a reference-free statistical framework for inferring population admixture history from identity-by-descent (IBD) segments. Based on the coalescent process under the Hybrid Isolation (HI) model, AdmixIBD exploits the effects of population divergence and admixture on genome-wide IBD patterns to infer demographic parameters without requiring predefined ancestral reference populations.

The framework integrates two complementary summaries of IBD: the **IBD-sharing ratio**, which primarily informs effective population size and admixture proportion, and the **conditional length distribution of detectable IBD segments**, which is particularly informative about admixture time. Through an alternating inference procedure, AdmixIBD iteratively estimates these demographic parameters using IBD segments detected within the target admixed population.

## Installation

AdmixIBD requires Python 3 and the following packages:

- `numpy`
- `scipy`
- `nlopt`

Clone the repository and install the dependencies:

```bash
git clone https://github.com/HuanyuLIN01/AdmixIBD.git
cd AdmixIBD
python -m pip install -r requirements.txt
```



## Quick Start

An example dataset and configuration file are provided in the `examples/` directory.

Run AdmixIBD from the repository root:

```bash
python AdmixIBD.py \
    --par-path examples/input_par.json \
    --result-path examples/new_result.json
```

The program reads the model configuration, loads the specified IBD segment lengths, performs parameter estimation, and saves the results to `examples/new_result.json`.

## Input

AdmixIBD requires two types of input: **IBD segment lengths** and **model configuration parameters**.

### 1. IBD Segment Lengths

The input file `examples/input.IBDlength_cM.txt` contains one IBD segment length per line, measured in centimorgans (cM).

### 2. Model Configuration

The file `examples/input_par.json` specifies the input data path and parameters used for inference.


| Parameter       | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `data_path`     | Path to the IBD segment length file                               |
| `ibd_label`     | Observed IBD-sharing ratio                                        |
| `N_hap_bound`   | Bounds for the effective population size parameter                |
| `tau_a_bound`   | Bounds for the admixture time parameter                           |
| `m_bound`       | Bounds for the admixture proportion parameter                     |
| `init_par`      | Initial parameter values, ordered as `N_hap`, `tau_a`, and `m`    |
| `fix`           | Whether each parameter is estimated (`0`) or fixed (`1`)          |
| `fix_val`       | Values assigned to fixed parameters                               |
| `gamma`, `beta` | Weights for the IBD-sharing and segment-length distribution terms |
| `rep`           | Replicate setting; use `[1]` for the supplied example             |




## Output

AdmixIBD writes the inference results to a JSON file at the path specified by `--result-path`.

The output includes estimated demographic parameters and optimization diagnostics from successive inference rounds.

An example output is provided at:

```text
examples/result.json
```

Some output field names retain their original spelling (e.g., `aproximated`) for consistency with the current implementation.

## Application to the 1000 Genomes Project

The `KGP/` directory contains supplementary input data and inference results from the real-data applications presented in the manuscript.

These include:

- **IBD segment lengths:** IBD data used for demographic inference.
- **IBD-sharing statistics:** Population-level IBD-sharing measurements.
- **Admixture proportions:** Population-level admixture estimates used in the analyses.
- **Inference results:** Results of applying AdmixIBD to the analyzed populations.

These materials are provided to support the interpretation and reproducibility of the real-data analyses.

## Citation

A manuscript describing AdmixIBD is currently in preparation. Citation information will be updated upon publication.