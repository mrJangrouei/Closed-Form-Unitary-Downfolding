# Closed-Form Unitary Downfolding (CFU) + LP-BLISS for Low-Norm Fermionic Hamiltonians

This repository contains a compact research implementation of the **Closed-Form Unitary (CFU) downfolding** framework and the **Linear-Programming Block-Invariant Symmetry Shift (LP-BLISS)**, developed to construct *spectrally compressed, low–1-norm Hamiltonians* for resource-efficient quantum simulation.

This code accompanies:

**M. R. Jangrouei**  
*Closed-Form Unitary Downfolding for Resource-Efficient Fermionic Hamiltonian Simulation on Quantum Computers.*

---

## 🔍 Overview

Fault-tolerant quantum algorithms such as **Qubitized Quantum Phase Estimation (QPE)** require expressing the electronic Hamiltonian as a **Linear Combination of Unitaries (LCU)**. The **LCU 1-norm** directly determines the query complexity and therefore the quantum resources required.

This repository implements a two-stage compression pipeline:

### **1. Closed-Form Unitary (CFU) Downfolding**
- Partitions orbitals into **internal** (active) and **external** subspaces.  
- Applies analytically solvable unitary transformations using closed-form BCH expansions.  
- Iteratively suppresses internal–external entanglement.  
- Constructs a spectrally compressed, active-space effective Hamiltonian \( \hat{H}_{\mathrm{eff}} \).  
- Controls operator growth using Frobenius-norm–bounded pruning.

### **2. LP-BLISS LCU Minimization**
- Optimizes a number-conserving BLISS operator via **linear programming**:
  \[
      \min_{\theta} \|\mathbf{b} - A\theta\|_1
  \]
- Achieves **significant LCU 1-norm reduction** while preserving the spectrum in the desired electron-number sector.  
- Pushes beyond traditional spectral-range limits when combined with CFU.

---

## 📁 File Structure

This implementation is intentionally lightweight:

