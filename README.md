# Closed-Form Unitary Downfolding (CFU) + LP-BLISS for Low-Norm Fermionic Hamiltonians

This repository contains a compact research implementation of **Closed-Form Unitary (CFU) downfolding**, **Linear-Programming Block-Invariant Symmetry Shift (LP-BLISS)**, and **active-space Hamiltonian construction** for resource-efficient fermionic quantum simulation.

The code accompanies:

**M. R. Jangrouei**  
*Closed-Form Unitary Downfolding for Resource-Efficient Fermionic Hamiltonian Simulation on Quantum Computers.*

---

## What This Repository Does

Fault-tolerant algorithms such as **qubitized quantum phase estimation (QPE)** require the electronic Hamiltonian to be represented as a **linear combination of unitaries (LCU)**. The LCU 1-norm strongly affects query complexity and therefore quantum resource estimates.

This repository explores ways to compress fermionic Hamiltonians before quantum simulation. The main ingredients are:

1. **Closed-form unitary transformations** to reduce undesired couplings.
2. **LP-BLISS symmetry shifts** to reduce the Hamiltonian 1-norm without changing the target electron-number-sector spectrum.
3. **Frobenius-norm pruning** to remove small terms with controlled discarded norm.
4. **Active-space downfolding** to project the transformed Hamiltonian into a smaller internal orbital space.

The final goal is to obtain a compact active-space Hamiltonian with reduced term count, reduced coefficient 1-norm, and preserved physically relevant spectral information.

---

## Quick Start

A typical workflow has two stages.

### Stage 1: Choose one transformation script

Run **one** of the following transformation pipelines:

| Option | Script | What it does |
|---|---|---|
| A | `cfu_transform.py` | Baseline closed-form unitary transformation. |
| B | `cfu_lp_bliss.py` | CFU transformation plus LP-BLISS compression. |
| C | `cfu_lp_bliss_frobenius.py` | CFU plus LP-BLISS plus Frobenius-norm pruning. |

For example:

```bash
python cfu_transform.py
```

or

```bash
python cfu_lp_bliss.py
```

or

```bash
python cfu_lp_bliss_frobenius.py
```

Each script produces transformed full-space Hamiltonians such as:

```text
hamiltonian_ST_minE_transformed-1.pkl
hamiltonian_ST_minE_transformed-2.pkl
...
hamiltonian_ST_minE_transformed-k.pkl
```

### Stage 2: Downfold the transformed Hamiltonian

After generating a transformed Hamiltonian, pass it to the downfolding script:

```bash
python downfold.py
```

The downfolding script projects out the external spin orbitals and saves the active-space Hamiltonian as:

```text
downfolded_hamiltonian.pkl
```

The workflow is therefore:

```text
Input full-space Hamiltonian
        |
        v
Choose one transformation pipeline
        |
        v
Transformed full-space Hamiltonian
        |
        v
Downfolding / external HF projection
        |
        v
Active-space effective Hamiltonian
```

---

## Repository Layout

| File | Purpose |
|---|---|
| `paper.tex` | Manuscript and theory draft. |
| `cfu_transform.py` | Baseline CFU transformation. |
| `cfu_lp_bliss.py` | CFU transformation with LP-BLISS compression. |
| `cfu_lp_bliss_frobenius.py` | CFU transformation with LP-BLISS and Frobenius-norm pruning. |
| `downfold.py` | Projects a transformed full-space Hamiltonian into an active-space Hamiltonian. |
| `hamiltonian_ST.pkl` | Input full-space OpenFermion `FermionOperator`. |
| `hamiltonian_ST_minE_transformed-k.pkl` | Transformed Hamiltonian after `k` unitary steps. |
| `downfolded_hamiltonian.pkl` | Final active-space downfolded Hamiltonian. |

---

## Method Overview

The transformation scripts generate a full-space transformed Hamiltonian,

```math
\bar{H} = U H U^\dagger,
```

where `U` is built from a sequence of closed-form unitary rotations.

The downfolding script then constructs an active-space Hamiltonian by contracting the external spin orbitals with a fixed Hartree-Fock occupation pattern:

```math
H_{\mathrm{eff}}
=
\langle \phi_{\mathrm{ext}}^0 |
\bar{H}
| \phi_{\mathrm{ext}}^0 \rangle_{\mathrm{ext}}.
```

The resulting `H_eff` acts only on the selected internal spin orbitals.

---

## Transformation Pipelines

The three transformation scripts form a hierarchy:

```text
Baseline CFU
    -> CFU + LP-BLISS
        -> CFU + LP-BLISS + Frobenius truncation
```

### 1. Baseline CFU: `cfu_transform.py`

This is the core closed-form unitary transformation script.

It performs the iterative update

```math
H_{k+1}
=
e^{\theta_k A_k}
H_k
e^{-\theta_k A_k},
```

where `A_k` is an anti-Hermitian fermionic generator and `theta_k` is obtained by an analytic line search.

Main features:

- Loads a Hamiltonian from `hamiltonian_ST.pkl`.
- Uses OpenFermion `FermionOperator` objects.
- Uses a Hartree-Fock determinant as the reference state.
- Builds a generator pool from Hamiltonian terms.
- Includes single and double excitation generators.
- Excludes generators acting only inside the chosen internal orbital set.
- Selects the generator with the largest energy-gradient magnitude.
- Applies the closed-form similarity transformation.
- Optionally applies rank-based `n`-body truncation.
- Saves the transformed Hamiltonian after each unitary step.

Typical settings:

```python
HAM_PKL = Path("hamiltonian_ST.pkl")
ELEC_NUM = 4
INTERNAL_SPINORBS = {2, 3, 4, 5}
NumberOfUnitaries = 200

TRUNCATION_N_BODY = None
TRUNCATION_NUMBER_CONSERVING = False
```

Use this version to isolate the effect of the closed-form unitary transformations alone.

---

### 2. CFU + LP-BLISS: `cfu_lp_bliss.py`

This version adds LP-BLISS compression to the baseline CFU workflow.

The pipeline is:

```text
Initial LP-BLISS shift
        |
        v
CFU transformation step
        |
        v
Optional rank truncation
        |
        v
LP-BLISS shift
        |
        v
Next iteration
```

LP-BLISS shifts the Hamiltonian by an operator that vanishes in the target electron-number sector:

```math
H_{\mathrm{BLISS}}
=
H - O(\hat{N} - N_e \hat{I}).
```

Since

```math
(\hat{N} - N_e \hat{I}) |\Psi_{N_e}\rangle = 0,
```

this shift preserves the spectrum in the `N_e`-electron sector while changing the Hamiltonian coefficients.

The LP-BLISS optimization solves an L1 minimization problem of the form

```math
\min_{\theta} \|\mathbf{b} - A\theta\|_1,
```

with a support-preserving constraint so that no new Hamiltonian terms are introduced.

Main additions relative to baseline CFU:

- Initial BLISS shift before the CFU loop.
- BLISS shift after every CFU transformation.
- Number-operator symmetry channel `N - N_e I`.
- Sparse L1 linear programming using SciPy HiGHS.
- No-new-terms constraint.
- Explicit Hartree-Fock singles added to the generator pool.

Typical settings:

```python
HAM_PKL = Path("hamiltonian_ST.pkl")
ELEC_NUM = 4
INTERNAL_SPINORBS = {2, 3, 4, 5}
NumberOfUnitaries = 11

BLISS_TAIL_L1 = 0.0
BLISS_EPS_HELP = 1e-12
```

Use this version to study how much LCU 1-norm reduction is gained by symmetry-based BLISS compression.

---

### 3. CFU + LP-BLISS + Frobenius Truncation: `cfu_lp_bliss_frobenius.py`

This is the most complete transformation pipeline. It includes CFU transformations, LP-BLISS compression, and Frobenius-norm-controlled pruning.

The pipeline is:

```text
Initial LP-BLISS shift
        |
        v
CFU transformation step
        |
        v
Optional rank truncation
        |
        v
LP-BLISS shift
        |
        v
Frobenius-norm truncation
        |
        v
Next iteration
```

After each BLISS step, the script removes small terms while enforcing a bound on the discarded operator:

```math
\|\Delta H\|_F \le \epsilon.
```

Supported truncation modes:

| Mode | Description |
|---|---|
| `none` | Disable Frobenius truncation. |
| `linear` | Conservative triangle-inequality bound on dropped term norms. |
| `quadratic` | Tighter estimate of the full dropped-operator Frobenius norm. |

Typical settings:

```python
HAM_PKL = Path("hamiltonian_ST.pkl")
ELEC_NUM = 4
INTERNAL_SPINORBS = {2, 3, 4, 5}
NumberOfUnitaries = 11

BLISS_TAIL_L1 = 0.0
BLISS_EPS_HELP = 1e-12

FROB_TRUNCATION_MODE = "quadratic"
FROB_ENERGY_BUDGET_mH = 1.6
FROB_KEEP_CONSTANT = True
FROB_ORDER_BY = "beta0_mass"
```

Use this version to generate the most compact Hamiltonians for downstream resource estimates.

---

## Downfolding: `downfold.py`

The downfolding script is used **after** one of the three transformation scripts.

It loads a transformed full-space Hamiltonian and projects out selected external spin orbitals by contracting them with a fixed external Hartree-Fock state. The result is a smaller active-space Hamiltonian acting only on the internal spin orbitals.

The downfolding operation is

```math
H_{\mathrm{eff}}
=
\langle \phi_{\mathrm{ext}}^0 |
H_{\mathrm{transformed}}
| \phi_{\mathrm{ext}}^0 \rangle_{\mathrm{ext}}.
```

### Example orbital partition

```python
downfold_n_electrons = 2

EXTERNAL_ORBS_DF = {0, 1, 6, 7}
HF_EXTERNAL_OCC = [0, 1]
INTERNAL_ORBS_DF = {2, 3, 4, 5}
```

This means:

| Quantity | Value | Meaning |
|---|---|---|
| `EXTERNAL_ORBS_DF` | `{0, 1, 6, 7}` | Spin orbitals projected out. |
| `HF_EXTERNAL_OCC` | `[0, 1]` | Occupied external orbitals in the HF reference. |
| `INTERNAL_ORBS_DF` | `{2, 3, 4, 5}` | Spin orbitals retained in the active space. |

### Input

Set the transformed Hamiltonian filename inside `downfold.py`:

```python
filename = "hamiltonian_ST_minE_transformed-200.pkl"
```

This file may come from any transformation pipeline:

```text
cfu_transform.py
cfu_lp_bliss.py
cfu_lp_bliss_frobenius.py
```

### Output

The script saves the active-space Hamiltonian as:

```text
downfolded_hamiltonian.pkl
```

The output is renumbered so that the internal orbitals become contiguous. For example,

```text
2 -> 0
3 -> 1
4 -> 2
5 -> 3
```

so a Hamiltonian on internal spin orbitals `{2, 3, 4, 5}` becomes a Hamiltonian on local spin orbitals `{0, 1, 2, 3}`.

### Optional diagonalization

To diagonalize the downfolded Hamiltonian in a fixed electron-number sector, set:

```python
diagonalize_downfold = True
downfold_n_electrons = 2
```

The script then prints:

```text
Ground-state energy
Maximum energy
Spectral range
```

---

## Recommended Workflow

### 1. Prepare the input Hamiltonian

Save the full-space OpenFermion Hamiltonian as:

```text
hamiltonian_ST.pkl
```

### 2. Run one transformation pipeline

Choose one of:

```bash
python cfu_transform.py
python cfu_lp_bliss.py
python cfu_lp_bliss_frobenius.py
```

### 3. Select the transformed Hamiltonian

For example:

```text
hamiltonian_ST_minE_transformed-200.pkl
```

or, for shorter runs:

```text
hamiltonian_ST_minE_transformed-11.pkl
```

### 4. Update `downfold.py`

Edit:

```python
filename = "hamiltonian_ST_minE_transformed-200.pkl"
```

### 5. Run downfolding

```bash
python downfold.py
```

This generates:

```text
downfolded_hamiltonian.pkl
```

### 6. Use the active-space Hamiltonian

The final downfolded Hamiltonian can be used for:

- active-space exact diagonalization,
- quantum simulation,
- qubitization resource estimation,
- LCU 1-norm analysis,
- QPE cost estimates,
- comparison with FCI or CASCI benchmarks.

---

## Installation

Required packages:

```bash
pip install numpy scipy joblib openfermion matplotlib
```

Optional packages for generating molecular Hamiltonians:

```bash
pip install pyscf openfermionpyscf
```

---

## Implementation Notes

This is a research codebase intended for method development and numerical experimentation.

The scripts prioritize transparency and direct control over packaging. Most user options are defined near the top of each script.

The implementation uses:

- OpenFermion `FermionOperator` objects,
- sparse matrix-free expectation values,
- analytic single-generator line search,
- joblib-based parallelism,
- SciPy linear programming for BLISS,
- optional Frobenius-norm pruning,
- optional sparse diagonalization for downfolded Hamiltonians.

---

## Citation

If you use this code, please cite:

```text
M. R. Jangrouei,
Closed-Form Unitary Downfolding for Resource-Efficient Fermionic Hamiltonian Simulation on Quantum Computers.
```

---

## Disclaimer

This repository is under active research development. Interfaces, file names, and default parameters may change as the method is refined.
