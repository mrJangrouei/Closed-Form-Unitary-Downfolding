#!/usr/bin/env python3
"""
Generate and pickle the *full* STO‑3G Hamiltonian of H₂
(includes the nuclear‑repulsion constant).

Output
------
hamiltonian_original.pkl   –  normal‑ordered FermionOperator
"""

import pickle
from openfermion import (
    MolecularData,
    get_fermion_operator,
    normal_ordered
)
from openfermionpyscf import run_pyscf

# ────────── user parameters ──────────
BOND_LENGTH  =  1.0 * 1.735328 * 0.529177  # Å
BASIS        = "sto-3g"
CHARGE       = 0
MULTIPLICITY = 1       # singlet

def build_h2_hamiltonian(r_angs: float):
    """
    Return the *full* normal‑ordered FermionOperator for H₂
    at bond length r_angs (Å).  No energy shifts are applied.
    """
    geometry = [
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, r_angs)),
	    ("H", (0.0, 0.0, 2 * r_angs)),
	    ("H", (0.0, 0.0, 3 * r_angs))
    ]

    molecule = MolecularData(geometry, BASIS, MULTIPLICITY, CHARGE)
    molecule = run_pyscf(molecule,
                         run_scf=True,
                         run_mp2=False, run_cisd=False,
                         run_ccsd=False, run_fci=False)

    # Sanity prints
    print(f"Nuclear repulsion     : {molecule.nuclear_repulsion:.12f} Ha")
    print(f"PySCF total HF energy : {molecule.hf_energy:.12f} Ha")

    # Convert to second‑quantised form.
    H_op = get_fermion_operator(molecule.get_molecular_hamiltonian())

    # Constant term is already present; verify:
    const = H_op.terms.get((), 0.0)
    if abs(const - molecule.nuclear_repulsion) > 1e-8:
        print("WARNING: constant term mismatch!  Adding it manually.")
        H_op += get_fermion_operator({}, constant=molecule.nuclear_repulsion)

    return normal_ordered(H_op)

if __name__ == "__main__":
    H2 = build_h2_hamiltonian(BOND_LENGTH)

    with open("hamiltonian_ST.pkl", "wb") as f:
        pickle.dump(H2, f)

    print(f"Constant term stored  : {H2.terms.get((), 0.0):.12f} Ha")
    print("✓ Saved Hamiltonian →  hamiltonian_ST.pkl")

