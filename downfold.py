
#!/usr/bin/env python3
"""
Downfolding Code (No PySCF Calculation)
Loads a pre-transformed Hamiltonian (as a FermionOperator) from
"hamiltonian_reflection_transformed.pkl" and performs the downfolding procedure.

Naming convention:
  - EXTERNAL_ORBS_DF : the spin–orbital set used for the HF-projection
  - INTERNAL_ORBS_DF : complement in the full spin–orbital space (downfold target)
"""

import sys
import importlib.util
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

###############################################################################
# GLOBAL PARAMETERS FOR DOWNFOLDING
###############################################################################
downfold_n_electrons = 2        # how many electrons to place in the downfolded subspace
# The user-specified set of spin orbitals used for HF projection:
EXTERNAL_ORBS_DF = {0,1,6,7}  # external orbitals for HF projection
# Among those, specify which are occupied in the HF reference:
HF_EXTERNAL_OCC = [0,1]
# The internal orbitals set (the target for downfolding) is explicitly defined:
INTERNAL_ORBS_DF = {2,3,4,5}  # internal orbitals for downfolding

# Whether or not to diagonalize the resulting downfolded Hamiltonian
diagonalize_downfold = False

###############################################################################
# OPENFERMION IMPORTS
###############################################################################
from openfermion import (
    FermionOperator,
    normal_ordered,
    hermitian_conjugated,
    get_number_preserving_sparse_operator
)

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def build_external_hf_wavefunction(occupied_external_orbs, EXTERNAL_ORB_TO_LOCAL):
    """
    Build an HF wavefunction (as a dict {bitstring_id: amplitude=1.0})
    over the 'external' subspace orbitals. Each bit in the bitstring_id
    represents whether an orbital in EXTERNAL_ORBS_DF is occupied (1) or not (0).
    """
    state_id = 0
    for orb in occupied_external_orbs:
        local_idx = EXTERNAL_ORB_TO_LOCAL[orb]
        state_id |= (1 << local_idx)
    return {state_id: 1.0}

def apply_creation_op_external(wf, orb, EXTERNAL_ORB_TO_LOCAL):
    """Apply fermionic creation on 'orb' if orb is external; else do nothing."""
    if orb not in EXTERNAL_ORB_TO_LOCAL:
        return dict(wf)
    local_j = EXTERNAL_ORB_TO_LOCAL[orb]
    new_wf = {}
    for st_id, coeff in wf.items():
        if (st_id >> local_j) & 1:
            continue
        new_st = st_id | (1 << local_j)
        parity = sum((st_id >> k) & 1 for k in range(local_j))
        sign = (-1)**parity
        new_wf[new_st] = new_wf.get(new_st, 0.0) + coeff * sign
    return new_wf

def apply_annihilation_op_external(wf, orb, EXTERNAL_ORB_TO_LOCAL):
    """Apply fermionic annihilation on 'orb' if orb is external; else do nothing."""
    if orb not in EXTERNAL_ORB_TO_LOCAL:
        return dict(wf)
    local_j = EXTERNAL_ORB_TO_LOCAL[orb]
    new_wf = {}
    for st_id, coeff in wf.items():
        if not ((st_id >> local_j) & 1):
            continue
        new_st = st_id & ~(1 << local_j)
        parity = sum((st_id >> k) & 1 for k in range(local_j))
        sign = (-1)**parity
        new_wf[new_st] = new_wf.get(new_st, 0.0) + coeff * sign
    return new_wf

def overlap_internal(wfA, wfB):
    """Dot product of two wavefunctions in dict form."""
    return sum(ampA * wfB.get(st_id, 0.0) for st_id, ampA in wfA.items())

def fermion_swap_sign_df(opA, opB):
    """Whenever two fermionic operators are swapped, we multiply by -1."""
    return -1.0

def reorder_external_left(op_list, INTERNAL_ORBS_DF):
    """
    Reorder operators so that all operators on 'external' orbits come first,
    while operators on 'internal' orbits go last. Returns (overall_sign, new_op_list).
    """
    ops = list(op_list)
    sign = 1.0
    n = len(ops)
    def is_internal(orb):
        return orb in INTERNAL_ORBS_DF
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if is_internal(ops[j][1]) and not is_internal(ops[j+1][1]):
                ops[j], ops[j+1] = ops[j+1], ops[j]
                sign *= fermion_swap_sign_df(ops[j], ops[j+1])
    return sign, ops

def of_term_to_list_df(term_tuple):
    """
    Convert an OpenFermion term into a list of (op_type, orb).
    """
    op_list = []
    for (orb, action) in term_tuple:
        op_list.append(("C_dagger", orb) if action == 1 else ("C", orb))
    return op_list

def leftover_ops_to_of_df(ops, coeff):
    """
    Convert leftover internal operators back into a FermionOperator.
    """
    if not ops:
        return FermionOperator((), coeff)
    tokens = []
    for (op_type, orb) in ops:
        tokens.append(f"{orb}^" if op_type == "C_dagger" else f"{orb}")
    return FermionOperator(" ".join(tokens), coeff)

###############################################################################
# MAIN DOWNFOLDING PROCEDURE
###############################################################################
def main_downfold(transformed_hamiltonian):
    print(f"Transformed Hamiltonian has {len(transformed_hamiltonian.terms)} terms.")

    all_spin_orbs = set(range(16))
    global EXTERNAL_ORBS_DF, INTERNAL_ORBS_DF
    if INTERNAL_ORBS_DF is None:
        INTERNAL_ORBS_DF = all_spin_orbs - EXTERNAL_ORBS_DF

    print("=== Downfolding sets ===")
    print("EXTERNAL_ORBS_DF =", sorted(EXTERNAL_ORBS_DF))
    print("INTERNAL_ORBS_DF =", sorted(INTERNAL_ORBS_DF))

    EXTERNAL_ORB_TO_LOCAL = {orb: idx for idx, orb in enumerate(sorted(EXTERNAL_ORBS_DF))}
    hf_external = build_external_hf_wavefunction(HF_EXTERNAL_OCC, EXTERNAL_ORB_TO_LOCAL)

    downfolded_hamiltonian = FermionOperator()
    for term_ops, term_coeff in transformed_hamiltonian.terms.items():
        op_list = of_term_to_list_df(term_ops)
        reorder_sign, reordered_ops = reorder_external_left(op_list, INTERNAL_ORBS_DF)

        wf = dict(hf_external)
        for (op_type, orb) in reversed(reordered_ops):
            if orb in EXTERNAL_ORB_TO_LOCAL:
                wf = apply_creation_op_external(wf, orb, EXTERNAL_ORB_TO_LOCAL) if op_type == "C_dagger" else \
                     apply_annihilation_op_external(wf, orb, EXTERNAL_ORB_TO_LOCAL)

        scale = term_coeff * reorder_sign
        for st_id in wf:
            wf[st_id] *= scale
        val = overlap_internal(hf_external, wf)

        internal_ops = [(typ, o) for (typ, o) in reordered_ops if o in INTERNAL_ORBS_DF]
        term_op = leftover_ops_to_of_df(internal_ops, val)
        downfolded_hamiltonian += term_op

    downfolded_hamiltonian = normal_ordered(downfolded_hamiltonian)

    new_mapping = {old: new for new, old in enumerate(sorted(INTERNAL_ORBS_DF))}
    renumbered_op = FermionOperator()
    for term, coeff in downfolded_hamiltonian.terms.items():
        new_term = tuple((new_mapping[orb], action) for (orb, action) in term)
        renumbered_op += FermionOperator(new_term, coeff)

    if diagonalize_downfold:
        n_downfold_spin = len(new_mapping)
        sparse_op = get_number_preserving_sparse_operator(
            renumbered_op,
            n_downfold_spin,
            downfold_n_electrons,
            spin_preserving=False
        )
        dim = sparse_op.shape[0]
        print(f"\nDownfolded Hamiltonian dimension = {dim}")
        if dim <= 2:
            dense_mat = sparse_op.toarray()
            eigvals = np.linalg.eigvals(dense_mat)
            e_min_down, e_max_down = np.min(eigvals), np.max(eigvals)
        else:
            e_min_down = eigs(sparse_op, k=1, which='SR', return_eigenvectors=False)[0]
            e_max_down = eigs(sparse_op, k=1, which='LR', return_eigenvectors=False)[0]

        print("\n=== Downfolded Hamiltonian Diagonalization ===")
        print(f"Number of electrons in downfolded space = {downfold_n_electrons}")
        print(f"Number of spin orbitals in downfolded space = {n_downfold_spin}")
        print(f"Ground State Energy (E_min) = {e_min_down}")
        print(f"Maximum Energy (E_max)     = {e_max_down}")
        print(f"Spectral Range = {e_max_down - e_min_down}")
    else:
        print("\nDiagonalization of Downfolded Hamiltonian is turned off.")

    # Save the renumbered downfolded Hamiltonian directly
    with open("downfolded_hamiltonian.pkl", "wb") as f:
        pickle.dump(renumbered_op, f)
    print("\nRenumbered downfolded Hamiltonian (FermionOperator) saved to 'downfolded_hamiltonian.pkl'.")

###############################################################################
# MAIN
###############################################################################
def main():
    filename = "hamiltonian_ST_minE_transformed-200.pkl"
    print(f"Loading UCC-transformed Hamiltonian from '{filename}'...")
    with open(filename, "rb") as f:
        reflection_transformed_hamiltonian = pickle.load(f)
    main_downfold(reflection_transformed_hamiltonian)

if __name__ == "__main__":
    main()
