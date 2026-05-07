# Closed-Form Unitary Downfolding (CFU) + LP-BLISS for Low-Norm Fermionic Hamiltonians

This repository contains a compact research implementation of **Closed-Form Unitary (CFU) downfolding**, **Linear-Programming Block-Invariant Symmetry Shift (LP-BLISS)**, and active-space Hamiltonian construction for resource-efficient quantum simulation of fermionic systems.

The code accompanies:

**M. R. Jangrouei**  
*Closed-Form Unitary Downfolding for Resource-Efficient Fermionic Hamiltonian Simulation on Quantum Computers.*

---

## Overview

Fault-tolerant quantum algorithms such as **qubitized quantum phase estimation (QPE)** require expressing the electronic Hamiltonian as a **linear combination of unitaries (LCU)**. The LCU 1-norm strongly affects the query complexity and therefore the quantum resources required.

This repository implements and compares several Hamiltonian-compression strategies based on:

1. **Closed-form unitary transformations**
2. **LP-BLISS symmetry-preserving LCU minimization**
3. **Frobenius-norm-controlled pruning**
4. **External-space projection / active-space downfolding**

The overall goal is to construct compact effective Hamiltonians that preserve the relevant low-energy physics while reducing operator norm, term count, and simulation cost.

---

## Repository Components

The repository contains five main components:

```text
1. Manuscript / theory draft
2. Baseline CFU transformation code
3. CFU + LP-BLISS code
4. CFU + LP-BLISS + Frobenius truncation code
5. Downfolding / active-space projection code
