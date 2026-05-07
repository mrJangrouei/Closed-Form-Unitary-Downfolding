#!/usr/bin/env python3
"""
Steepest-gradient single-generator similarity transform (iterated, fast & screened)
— Matrix-free + joblib-parallel version — + LP-BLISS shift

Pipeline:
    1. Load original Hamiltonian H_raw from HAM_PKL.
    2. (Optional) prune tiny terms.
    3. Run an initial LP-BLISS shift in the (N̂ − Nₑ·I) channel:
           H_raw  →  H0 = bliss_shift(H_raw, N_e).
    4. For k = 1..NumberOfUnitaries:
           - Build generator pool (S, D) from current H_k
             + always include HF singles a_v† a_i − a_i† a_v.
           - Choose A_k by steepest |g(0)|.
           - Analytic θ* via quartic line search, apply similarity transform:
                 H_k → Ḣ_k = e^{θ_k A_k} H_k e^{-θ_k A_k}.
           - Optional ≤ n-body truncation (H_trunc).
           - LP-BLISS shift again:
                 H_{k+1} = bliss_shift(H_trunc, N_e).
           - Save H_{k+1}.

Features:
  • Matrix-free HF expectation using sparse bitstring states.
  • Analytic line search: solve quartic in t = tan(θ/2).
  • General n-body truncation control (none / number-conserving / non-conserving).
  • LP-BLISS:
      - Preimage O-basis from H (rank r−1).
      - Cached (T)·(N̂−Nₑ) and (T†)·(N̂−Nₑ).
      - Sparse L¹ LP with SciPy HiGHS.
      - No-new-terms constraint (support(H_shift) ⊆ support(H)).
  • HF singles pool:
      - Always include all HF singles excitations (subject to INTERNAL_SPINORBS),
        independent of the current Hamiltonian support.

Requires:
    - python ≥ 3.10 (for | in type hints) or adjust annotations manually.
    - openfermion ≥ 1.5
    - numpy
    - scipy (sparse + optimize.linprog)
    - joblib  (optional; falls back to threads)
"""

import os
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass

import numpy as np

# --- joblib (graceful fallback if missing) ---
try:
    from joblib import Parallel, delayed, parallel_backend  # type: ignore
    HAVE_JOBLIB = True
except ModuleNotFoundError:
    HAVE_JOBLIB = False
    from contextlib import contextmanager
    from concurrent.futures import ThreadPoolExecutor

    def delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    def Parallel(n_jobs=None, batch_size=None):
        def runner(callables):
            if n_jobs in (None, -1):
                max_workers = os.cpu_count() or 1
            elif isinstance(n_jobs, int) and n_jobs > 0:
                max_workers = n_jobs
            else:
                max_workers = 1
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(c) for c in callables]
                return [f.result() for f in futs]
        return runner

    from contextlib import contextmanager
    @contextmanager
    def parallel_backend(_name, inner_max_num_threads=None):
        yield

# --- OpenFermion ---
from openfermion import (
    FermionOperator,
    normal_ordered,
    hermitian_conjugated,
    count_qubits,
    number_operator,
)

# --- SciPy (required for BLISS) ---
try:
    import scipy.sparse as sp
    from scipy.optimize import linprog
except ImportError as e:
    raise ImportError("BLISS shift requires SciPy (e.g. `pip install scipy`).") from e

# ────────── USER SETTINGS ──────────
HAM_PKL      = Path("hamiltonian_ST.pkl")
ELEC_NUM     = 4
INTERNAL_SPINORBS: Set[int] = {2, 3, 4, 5}  # exclude generators whose indices are all inside this set
TOL          = 1e-10                        # α-test tolerance
SAVE_OUT     = True
OUT_PKL      = HAM_PKL.with_stem(HAM_PKL.stem + "_minE_transformed")
NumberOfUnitaries = 11                     # successive transforms

# Optional speed toggle
HAM_PRUNE_EPS = 0.0                         # e.g., 1e-12 to drop tiny H terms; 0.0 disables

# joblib controls
JOBLIB_BACKEND        = "loky"              # "loky" or "threads"
SCAN_N_JOBS           = -1                  # g(0) scan
COEFFS_N_JOBS         = -1                  # winner coeffs
TRANSFORM_N_JOBS      = -1                  # transform chunks
INNER_MAX_NUM_THREADS = 1                   # pin BLAS in workers

# Transform / truncation behavior
EARLY_TRUNCATE_DURING_TRANSFORM = False
TRUNCATION_N_BODY: Optional[int] = None     # e.g. 1,2,3,… or None for **no truncation**
TRUNCATION_NUMBER_CONSERVING     = False

# LP-BLISS controls
BLISS_TAIL_L1   = 0.0    # L1 mass of smallest-|coeff| tail used for elimination screening
BLISS_EPS_HELP  = 1e-12  # helpfulness threshold in Ne_diff response

# ────────── BASIC HELPERS ──────────
l1   = lambda op: sum(abs(c) for c in normal_ordered(op).terms.values())
comm = lambda X, Y: normal_ordered(X * Y - Y * X)
dcomm = lambda X, Y: comm(comm(X, Y), Y)

def alpha(A: FermionOperator, h: FermionOperator) -> int:
    """Evangelista α test: α∈{1,4} cases supported."""
    S = normal_ordered(A * comm(h, A) * A)
    if l1(S) <= TOL:
        return 1
    if l1(S - comm(h, A)) <= TOL:
        return 4
    raise RuntimeError("α-test failed (neither α=1 nor α=4).")

# ────────── Matrix-free Fock-basis utilities ──────────
SparseState = Dict[int, complex]  # basis index → amplitude

def popcount_below(x: int, p: int) -> int:
    if p <= 0:
        return 0
    return (x & ((1 << p) - 1)).bit_count()

def apply_term_to_basis(
    term: Tuple[Tuple[int, int], ...],
    bitidx: int,
    amp: complex = 1.0
) -> Tuple[int, complex] | None:
    """Apply monomial ladder string to |bitidx⟩. Rightmost acts first."""
    occ = bitidx
    phase = 1.0
    for p, a in reversed(term):
        parity = -1.0 if (popcount_below(occ, p) % 2) else 1.0
        if a == 0:  # annihilation
            if ((occ >> p) & 1) == 0:
                return None
            phase *= parity
            occ ^= (1 << p)
        else:       # creation
            if ((occ >> p) & 1) == 1:
                return None
            phase *= parity
            occ ^= (1 << p)
    return occ, amp * phase

def basis_vec(bitidx: int, amp: complex = 1.0) -> SparseState:
    return {bitidx: complex(amp)}

def apply_op_to_sparse(op: FermionOperator, vec: SparseState) -> SparseState:
    if not op.terms or not vec:
        return {}
    out: SparseState = {}
    op_terms = sorted(op.terms.items())  # deterministic
    vec_items = list(vec.items())
    for term, coeff in op_terms:
        if coeff == 0:
            continue
        for idx, amp in vec_items:
            res = apply_term_to_basis(term, idx, amp)
            if res is None:
                continue
            j, a = res
            val = a * coeff
            if val != 0:
                out[j] = out.get(j, 0.0) + val
                if out[j] == 0:
                    del out[j]
    return out

def vdot_sparse(x: SparseState, y: SparseState) -> complex:
    if not x or not y:
        return 0.0
    if len(x) <= len(y):
        keys = sorted(x.keys())
        return sum(np.conj(x[k]) * y.get(k, 0.0) for k in keys)
    else:
        keys = sorted(y.keys())
        return sum(np.conj(x.get(k, 0.0)) * y[k] for k in keys)

# ────────── g(θ) / E(θ) scalars ──────────

def g_theta(th, e11, e21, e14, e24):
    sinθ, cosθ = math.sin(th), math.cos(th)
    sin2, cos2 = math.sin(2 * th), math.cos(2 * th)
    return cosθ * e11 + sinθ * e21 + cos2 * e14 + 0.5 * sin2 * e24

def E_theta(th, e00, e11, e21, e14, e24):
    sinθ, cosθ = math.sin(th), math.cos(th)
    sin2 = math.sin(2 * th)
    return (
        e00
        + sinθ * e11
        + (1 - cosθ) * e21
        + 0.5 * sin2 * e14
        + 0.5 * (sinθ ** 2) * e24
    )

# ────────── Globals (lazy-initialized per worker) ──────────
GLOB: Dict[str, object] = {}
# Keys: H_terms, H_op, H_mask, n_qb, ket_sparse, bit_idx, Hket_sparse, _ident

@dataclass(frozen=True)
class Ctx:
    H_terms: List[Tuple[Tuple[Tuple[int, int], ...], complex]]
    H_op: FermionOperator
    H_mask: List[int]
    n_qb: int
    ket_sparse: SparseState
    bit_idx: int
    ident: Tuple[int, int, int]  # (epoch_k, len(H_terms), bit_idx)

def _init_worker(H_terms, H_op, H_mask, n_qb, ket_sparse, bit_idx):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    GLOB["H_terms"] = H_terms
    GLOB["H_op"]    = H_op
    GLOB["H_mask"]  = H_mask
    GLOB["n_qb"]    = n_qb
    GLOB["ket_sparse"]  = ket_sparse
    GLOB["bit_idx"]     = bit_idx
    GLOB["Hket_sparse"] = apply_op_to_sparse(H_op, ket_sparse)

def _ensure_ctx(ctx: Ctx):
    if GLOB.get("_ident") != ctx.ident:
        _init_worker(ctx.H_terms, ctx.H_op, ctx.H_mask, ctx.n_qb, ctx.ket_sparse, ctx.bit_idx)
        GLOB["_ident"] = ctx.ident

# ────────── HF reference (sparse only) ──────────

def canonical_hf_state(n_qb: int, n_elec: int):
    occ = tuple(range(n_elec))
    bit_idx = 0
    for q in occ:
        bit_idx |= (1 << q)
    ket = basis_vec(bit_idx, 1.0)
    return occ, bit_idx, ket

# ────────── HF-action & supports ──────────

def term_acts_nonzero_on_occ(term_tuple, occ_set):
    occ = set(occ_set)
    for p, a in reversed(term_tuple):
        if a == 0:
            if p not in occ:
                return False
            occ.remove(p)
        else:
            if p in occ:
                return False
            occ.add(p)
    return True

def op_acts_nonzero_on_HF(op: FermionOperator, occ_set):
    if not op.terms:
        return False
    term_tuple = next(iter(op.terms.keys()))
    if term_acts_nonzero_on_occ(term_tuple, occ_set):
        return True
    adj = hermitian_conjugated(op)
    adj_term = next(iter(adj.terms.keys()))
    return term_acts_nonzero_on_occ(adj_term, occ_set)

# ────────── FAST BITMASK SUPPORTS ──────────

def mask_from_term(term) -> int:
    m = 0
    for (p, _a) in term:
        m |= (1 << p)
    return m

def mask_from_A(A: FermionOperator) -> int:
    m = 0
    for t in A.terms.keys():
        for (p, _a) in t:
            m |= (1 << p)
    return m

# ────────── Generator pool (threaded build + dedup) ──────────
from concurrent.futures import ThreadPoolExecutor

def _gen_from_term(term, occ_set, internal_set: Set[int]):
    inds = {p for (p, _a) in term}
    # Original filter: drop generators whose indices are ALL inside internal_set
    if inds and inds.issubset(internal_set):
        return None
    T = FermionOperator(term, 1.0)
    X = hermitian_conjugated(T)
    if not op_acts_nonzero_on_HF(X, occ_set):
        return None
    ladd = next(iter(X.terms.keys()))
    creators  = tuple(p for (p, a) in ladd if a == 1)
    annihils  = tuple(p for (p, a) in ladd if a == 0)
    if   len(creators) == 1 and len(annihils) == 1:
        tag, id1, id2 = "S", creators[0], annihils[0]
    elif len(creators) == 2 and len(annihils) == 2:
        tag, id1, id2 = "D", tuple(creators), tuple(annihils)
    else:
        return None
    A = normal_ordered(X - hermitian_conjugated(X))
    if not A.terms:
        return None
    key = tuple(sorted(A.terms.items()))
    return (tag, id1, id2, A, key)

def build_generators_threaded(
    H_terms,
    occ_HF,
    internal_set: Set[int],
    max_workers: int
) -> List[Tuple[str, object, object, FermionOperator]]:
    occ_set = set(occ_HF)
    with ThreadPoolExecutor(max_workers=max_workers) as tpool:
        cands = tpool.map(lambda tc: _gen_from_term(tc[0], occ_set, internal_set), H_terms)
    unique: Dict[Tuple, Tuple[str, object, object, FermionOperator]] = {}
    for item in cands:
        if item is None:
            continue
        tag, id1, id2, A, key = item
        if key not in unique:
            unique[key] = (tag, id1, id2, A)
    return list(unique.values())

# ────────── NEW: HF-based singles generators (always present) ──────────

def build_hf_singles_generators(
    n_qb: int,
    occ_HF: Tuple[int, ...],
    internal_set: Set[int]
) -> List[Tuple[str, int, int, FermionOperator]]:
    """
    Build a fixed UCC-style singles pool from the HF occupation pattern,
    independent of whether those terms appear in H.

    For each occupied orbital i and virtual orbital a (with i != a):
      A_ai = a_a^\dagger a_i - a_i^\dagger a_a

    We exclude only those generators whose BOTH indices lie inside
    `internal_set`, to match the original INTERNAL_SPINORBS filtering
    used in _gen_from_term.
    """
    occ_set = set(occ_HF)
    virt = [p for p in range(n_qb) if p not in occ_set]

    singles: List[Tuple[str, int, int, FermionOperator]] = []
    for i in occ_set:
        for a in virt:
            if i == a:
                continue
            # drop only if {i,a} ⊆ internal_set
            if {i, a}.issubset(internal_set):
                continue
            X = FermionOperator(((a, 1), (i, 0)), 1.0)   # a_a† a_i
            A = normal_ordered(X - hermitian_conjugated(X))
            if not A.terms:
                continue
            # Convention: tag "S", id1=virtual (a), id2=occupied (i): S a->i
            singles.append(("S", a, i, A))
    return singles

# ────────── g(0) scan (joblib) ──────────

def _scan_g0_job(ctx: Ctx, gen):
    _ensure_ctx(ctx)
    tag, id1, id2, A = gen
    ket  = GLOB["ket_sparse"]
    Hket = GLOB["Hket_sparse"]
    psi  = apply_op_to_sparse(A, ket)
    g0   = 2.0 * float(np.real(vdot_sparse(Hket, psi)))
    return tag, id1, id2, A, g0

# ────────── Analytic θ* via quartic in t = tan(θ/2) ──────────

def _theta_star_via_roots(e00, e11, e21, e14, e24):
    coeffs = np.array(
        [e14 - e11, 2.0 * (e21 - e24), -6.0 * e14, 2.0 * (e21 + e24), e11 + e14],
        dtype=float,
    )
    nz = np.flatnonzero(np.abs(coeffs) > 1e-14)
    if nz.size == 0:
        ths = [0.0, math.pi]
    else:
        coeffs = coeffs[nz[0]:]
        roots = np.roots(coeffs)
        ts = [r.real for r in roots if abs(r.imag) < 1e-10]
        ths = [0.0, math.pi] + [2.0 * math.atan(t) for t in ts]
    norm = lambda th: (th + 2 * math.pi) % (2 * math.pi)
    cand = sorted({round(norm(th), 12) for th in ths})
    best = None
    for th in cand:
        val = E_theta(th, e00, e11, e21, e14, e24)
        if (best is None) or (val < best[1]):
            best = (th, val)
    th_star, E_star = best
    g_star = g_theta(th_star, e11, e21, e14, e24)
    th_star = (th_star + 2 * math.pi) % (2 * math.pi)
    return th_star, float(E_star), float(g_star)

# ────────── Winner coeffs (joblib across slices) ──────────

def _coeffs_chunk_job(
    ctx: Ctx,
    idx_slice: List[int],
    A: FermionOperator,
    ket: SparseState,
    psi: SparseState,
    psi2: SparseState,
    A_mask: int
):
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]
    H_mask  = GLOB["H_mask"]
    acc = {1: {'E1': 0.0, 'E2': 0.0}, 4: {'E1': 0.0, 'E2': 0.0}}
    for i in idx_slice:
        if (H_mask[i] & A_mask) == 0:
            continue
        term, ch = H_terms[i]
        h = FermionOperator(term, ch)
        a = alpha(A, h)
        phi  = apply_op_to_sparse(h,  ket)
        hpsi = apply_op_to_sparse(h,  psi)
        E1 = vdot_sparse(ket, hpsi) + vdot_sparse(psi,  phi)
        E2 = vdot_sparse(ket, apply_op_to_sparse(h, psi2)) \
             + 2.0 * vdot_sparse(psi, hpsi) \
             + vdot_sparse(psi2, phi)
        acc[a]['E1'] += float(np.real(E1))
        acc[a]['E2'] += float(np.real(E2))
    return acc

def optimize_single_generator(ctx: Ctx, gen, e00):
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]
    nT      = len(H_terms)
    ket     = GLOB["ket_sparse"]
    tag, id1, id2, A = gen

    psi  = apply_op_to_sparse(A, ket)
    psi2 = apply_op_to_sparse(A, psi)
    A_mask = mask_from_A(A)

    chunk  = max(1, nT // ((os.cpu_count() or 1) * 4))
    idxs   = list(range(nT))
    slices = [idxs[i:i + chunk] for i in range(0, nT, chunk)]

    coeff = {1: {'E1': 0.0, 'E2': 0.0}, 4: {'E1': 0.0, 'E2': 0.0}}
    with parallel_backend(JOBLIB_BACKEND, inner_max_num_threads=INNER_MAX_NUM_THREADS):
        parts = Parallel(n_jobs=COEFFS_N_JOBS, batch_size="auto")(
            delayed(_coeffs_chunk_job)(ctx, sl, A, ket, psi, psi2, A_mask)
            for sl in slices
        )
    for acc in parts:
        for k in (1, 4):
            coeff[k]['E1'] += acc[k]['E1']
            coeff[k]['E2'] += acc[k]['E2']

    e11, e21 = coeff[1]['E1'], coeff[1]['E2']
    e14, e24 = coeff[4]['E1'], coeff[4]['E2']

    θ, E, g = _theta_star_via_roots(e00, e11, e21, e14, e24)
    θ = (θ + 2 * math.pi) % (2 * math.pi)
    return tag, id1, id2, A, θ, E, g

# ────────── General ≤ n-body truncation ──────────

def _term_counts(term) -> Tuple[int, int]:
    creators = sum(1 for (_p, a) in term if a == 1)
    annihils = len(term) - creators
    return creators, annihils

def _keep_term(term, n_body: Optional[int], number_conserving: bool) -> bool:
    if len(term) == 0:
        return True
    if n_body is None:
        return True  # no truncation
    c, a = _term_counts(term)
    if number_conserving:
        if c != a:
            return False
        return c <= n_body  # creators == annihils == k ≤ n
    else:
        return max(c, a) <= n_body

def truncate_to_n_body(
    H_op: FermionOperator,
    n_body: Optional[int],
    number_conserving: bool = True
) -> FermionOperator:
    if n_body is None:
        return normal_ordered(H_op)
    Hn = FermionOperator()
    for term, ch in H_op.terms.items():
        if _keep_term(term, n_body, number_conserving):
            Hn += FermionOperator(term, ch)
    return normal_ordered(Hn)

# ────────── Transform (chunked) + optional early truncation, joblib ──────────

def transform(α: int, O: FermionOperator, A: FermionOperator, θ: float) -> FermionOperator:
    OA, OAA = comm(O, A), dcomm(O, A)
    if α == 1:
        return O + math.sin(θ) * OA + (1 - math.cos(θ)) * OAA
    return O + 0.5 * math.sin(2 * θ) * OA + 0.5 * (math.sin(θ) ** 2) * OAA

def _transform_chunk_job(
    ctx: Ctx,
    idx_slice: List[int],
    A: FermionOperator,
    θ: float,
    A_mask: int,
    early_trunc: bool,
    n_body: Optional[int],
    number_conserving: bool
) -> FermionOperator:
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]
    H_mask  = GLOB["H_mask"]
    acc = FermionOperator()
    for i in idx_slice:
        term, ch = H_terms[i]
        if (H_mask[i] & A_mask) == 0:
            acc += FermionOperator(term, ch)
            continue
        h = FermionOperator(term, ch)
        a = alpha(A, h)
        h_tr = transform(a, h, A, θ)
        if early_trunc:
            acc += truncate_to_n_body(h_tr, n_body, number_conserving)
        else:
            acc += h_tr
    return normal_ordered(acc)

def transform_streaming_joblib(
    ctx: Ctx,
    A: FermionOperator,
    θ: float,
    early_trunc: bool,
    n_body: Optional[int],
    number_conserving: bool
) -> FermionOperator:
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]
    nT      = len(H_terms)
    A_mask  = mask_from_A(A)
    chunk   = max(1, nT // ((os.cpu_count() or 1) * 4))
    idxs    = list(range(nT))
    slices  = [idxs[i:i + chunk] for i in range(0, nT, chunk)]

    with parallel_backend(JOBLIB_BACKEND, inner_max_num_threads=INNER_MAX_NUM_THREADS):
        parts = Parallel(n_jobs=TRANSFORM_N_JOBS, batch_size="auto")(
            delayed(_transform_chunk_job)(
                ctx, sl, A, θ, A_mask, early_trunc, n_body, number_conserving
            ) for sl in slices
        )

    # Pairwise reduce to keep peak small
    while len(parts) > 1:
        nxt: List[FermionOperator] = []
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                nxt.append(normal_ordered(parts[i] + parts[i + 1]))
            else:
                nxt.append(parts[i])
        parts = nxt
    return parts[0] if parts else FermionOperator()

# ────────── Expectations on HF (matrix-free) ──────────

def expect_on_HF(op: FermionOperator) -> float:
    ket = GLOB["ket_sparse"]
    y   = apply_op_to_sparse(op, ket)
    val = vdot_sparse(ket, y)
    return float(np.real(val))

# ────────── LP-BLISS shift (number-operator channel) ──────────

def is_creation_tag(tag) -> bool:
    """Robust creation tag check (works for 0/1 and string tags)."""
    if isinstance(tag, (int, np.integer)):
        return int(tag) != 0
    if isinstance(tag, str):
        return tag in ("^", "+", "1", "create", "Creation", "CREATION")
    return bool(tag)

def build_preimage_basis_from_H(terms_in_H):
    """
    For each term in H, delete one a† and one a in all ways; keep
    number-preserving results (drop identity). This produces rank-(r−1)
    "preimages" of the Hamiltonian terms.
    """
    basis_set = set()
    for t in terms_in_H:
        if not t:
            continue
        ops = list(t)
        cre_pos = [i for i, (_m, tag) in enumerate(ops) if is_creation_tag(tag)]
        ann_pos = [i for i, (_m, tag) in enumerate(ops) if not is_creation_tag(tag)]
        if not cre_pos or not ann_pos:
            continue
        for i in cre_pos:
            for j in ann_pos:
                uu = ops.copy()
                for k in sorted([i, j], reverse=True):
                    uu.pop(k)
                u = tuple(uu)
                if not u:
                    continue
                ccount = sum(1 for (_m, tag) in u if is_creation_tag(tag))
                if 2 * ccount == len(u):
                    basis_set.add(u)
    return sorted(basis_set)

def make_screening_support_small_tail(H: FermionOperator, tail_l1: float):
    """
    Build the 'screening' support of H:

    We collect the SMALLEST-|coeff| terms of H in ascending |coeff| order
    until their accumulated L¹ mass reaches at most `tail_l1`.
    These terms form H_elim_support and are the ONLY terms used in
    the elimination/helpfulness check. If `tail_l1` ≤ 0, or if no terms
    fit under the threshold, we fall back to using all terms.
    """
    if not H.terms:
        return set(), 0.0, 0
    if tail_l1 <= 0.0:
        # No special tail – use full H for screening
        return set(H.terms.keys()), 0.0, len(H.terms)

    items = sorted(
        ((t, abs(c)) for t, c in H.terms.items()),
        key=lambda x: x[1],
    )

    cum = 0.0
    kept_terms: List[Tuple[Tuple[Tuple[int, int], ...]]] = []
    for t, a in items:
        if cum + a <= tail_l1:
            kept_terms.append(t)
            cum += a
        else:
            break

    if not kept_terms:
        # Threshold smaller than smallest |coeff| → fallback to full H
        return set(H.terms.keys()), 0.0, 0

    support = set(kept_terms)
    return support, cum, len(kept_terms)

def build_H_from_residual(
    residual_r: np.ndarray,
    residual_i: np.ndarray,
    term2idx,
    support=None
) -> FermionOperator:
    """
    Build a FermionOperator from residual vectors (real + imag), optionally
    restricting to a given support set of terms.

    By construction, this guarantees supp(H_shifted) ⊆ support if provided.
    """
    H_new = FermionOperator()
    for t, idx in term2idx.items():
        if support is not None and t not in support:
            continue
        c_real = float(residual_r[idx])
        c_imag = float(residual_i[idx])
        if c_real == 0.0 and c_imag == 0.0:
            continue
        c = c_real + 1j * c_imag
        H_new += FermionOperator(t, c)
    return normal_ordered(H_new)

def solve_lp_l1_sparse(
    Dr: sp.csc_matrix,
    Di: sp.csc_matrix,
    Hr: np.ndarray,
    Hi: np.ndarray,
    new_term_indices: np.ndarray
) -> np.ndarray:
    """
    Solve min_θ ||Hr - Dr θ||₁ + ||Hi - Di θ||₁
    subject to Dr[new_term_indices] θ = 0 and Di[new_term_indices] θ = 0
    using a standard-form LP and SciPy's HiGHS backend.
    """
    n, P = Dr.shape  # n = number of rows (terms), P = number of parameters

    # L1 objective via positive/negative residual slacks:
    # For each row k:
    #   Hr_k = (Dr θ)_k + u^+_k - u^-_k,   u^+, u^- ≥ 0
    #   Hi_k = (Di θ)_k + v^+_k - v^-_k,   v^+, v^- ≥ 0
    # Objective: sum_k (u^+_k + u^-_k + v^+_k + v^-_k).

    w_r = np.ones(n, dtype=float)
    w_i = np.ones(n, dtype=float)

    # Decision vector: [θ (P), u_plus (n), u_minus (n), v_plus (n), v_minus (n)]
    c = np.concatenate([np.zeros(P), w_r, w_r, w_i, w_i])

    I  = sp.eye(n, format="csc")
    Zp = sp.csc_matrix((n, P))
    Zn = sp.csc_matrix((n, n))

    # Equalities for real and imaginary parts:
    #   Dr θ + u^+ - u^- = Hr
    #   Di θ + v^+ - v^- = Hi
    Aeq_top = sp.hstack([Dr,  I, -I, Zn, Zn], format="csc")
    Aeq_bot = sp.hstack([Di, Zn, Zn,  I, -I], format="csc")
    A_eq = sp.vstack([Aeq_top, Aeq_bot], format="csc")
    b_eq = np.concatenate([Hr, Hi])

    # Extra equalities for "no new terms": for each off-support row k,
    #   (Dr θ)_k = 0 and (Di θ)_k = 0.
    if new_term_indices.size > 0:
        DrN = Dr[new_term_indices, :]
        DiN = Di[new_term_indices, :]
        Aeq_extra = sp.vstack([DrN, DiN], format="csc")
        zeros_res = sp.csc_matrix((Aeq_extra.shape[0], 4 * n))  # no slack involvement
        A_eq = sp.vstack(
            [A_eq, sp.hstack([Aeq_extra, zeros_res], format="csc")],
            format="csc",
        )
        b_eq = np.concatenate([b_eq, np.zeros(Aeq_extra.shape[0])])

    # Bounds: θ free; slacks non-negative
    bounds = [(None, None)] * P + [(0, None)] * (4 * n)

    last_res = None
    for method in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=method,
            options={"presolve": True},
        )
        last_res = res
        if res.status == 0:
            return res.x[:P]

    raise RuntimeError(
        f"HiGHS linprog failed (status={last_res.status}, message={getattr(last_res, 'message', None)})"
    )

def bliss_shift(
    H: FermionOperator,
    n_e_target: int,
    n_orb: int,
    tail_l1: float = BLISS_TAIL_L1,
    eps_help: float = BLISS_EPS_HELP,
    verbose: bool = True
) -> FermionOperator:
    """
    Apply a BLISS-style LP shift in the (N̂ − Nₑ·I) channel to the FermionOperator H.

    Returns:
        H_shifted: FermionOperator with the same (or smaller) support as H.
    """
    H = normal_ordered(H)

    H_terms = len(H.terms)
    H_norm  = sum(abs(c) for c in H.terms.values())
    max_rank = max((len(t) for t in H.terms), default=0) // 2

    if verbose:
        print("\n[BLISS] ── LP-based shift on current Hamiltonian ──")
        print(f"[BLISS] H: terms = {H_terms}, 1-norm = {H_norm:.6e}, max rank = {max_rank}")

    # Number-operator channel: N̂ − n_e_target·I over n_orb modes
    Ne_diff = normal_ordered(number_operator(n_orb)) \
              - n_e_target * FermionOperator(())
    Ne_diff.compress(1e-8)

    # Tail support for elimination screening
    H_elim_support, tail_mass, tail_count = make_screening_support_small_tail(H, tail_l1)
    if verbose:
        print(
            "[BLISS] Elimination screening: using smallest-|coeff| tail with "
            f"total L1 = {tail_mass:.3e} over {tail_count} terms (threshold {tail_l1:.3e})."
        )
        print(f"[BLISS] Terms considered in elimination = {len(H_elim_support)} (H total {len(H.terms)})")

    # Preimage basis: rank-(r−1) preimages + identity
    basis_raw = {()}  # include identity explicitly
    basis_pre = build_preimage_basis_from_H(H.terms.keys())
    basis_raw |= set(basis_pre)
    if verbose:
        print(f"[BLISS] Preimage basis: {len(basis_pre)} non-identity monomials, "
              f"total {len(basis_raw)} including identity.")

    # Canonical reps (Hermitian parameterisation)
    canon_terms, is_self_adj, seen = [], [], set()
    for term in sorted(basis_raw):
        if term in seen:
            continue
        T  = FermionOperator(term, 1.0)
        Tc = normal_ordered(hermitian_conjugated(T))
        conj_key = next(iter(Tc.terms.keys()), term)
        if term == conj_key:
            canon_terms.append(term)
            is_self_adj.append(True)
            seen.add(term)
        else:
            rep = term if term < conj_key else conj_key
            canon_terms.append(rep)
            is_self_adj.append(False)
            seen.update({term, conj_key})

    index_map, idx = [], 0
    for flag in is_self_adj:
        if flag:
            index_map.append((idx, None))
            idx += 1
        else:
            index_map.append((idx, idx + 1))
            idx += 2
    param_count = idx
    if verbose:
        print(f"[BLISS] Operator basis (pre-screen): {len(canon_terms)} reps, "
              f"parameter vector length = {param_count}")

    # Cache per-term contributions & screen helpfulness
    cached_DT: List[Dict]   = []  # term -> complex for T·Ne_diff
    cached_DTc: List[Dict]  = []  # term -> complex for (T†)·Ne_diff
    helpful_mask: List[bool] = []

    def _compute_DT_DTc_and_helpful(term, self_flag):
        """Compute T·Ne_diff, (T†)·Ne_diff and the helpfulness flag for a single canonical term."""
        T_op = FermionOperator(term, 1.0)
        DT   = normal_ordered(T_op * Ne_diff).terms

        Tc_op = normal_ordered(hermitian_conjugated(T_op))
        DTc   = normal_ordered(Tc_op * Ne_diff).terms

        # Helpfulness: does α-impulse (self: DT, non-self: DT+DTc) overlap screening support?
        if self_flag:
            helpful = any(
                (t in H_elim_support)
                and (abs(c.real) > eps_help or abs(c.imag) > eps_help)
                for t, c in DT.items()
            )
        else:
            keys = (set(DT.keys()) | set(DTc.keys())) & H_elim_support
            helpful = False
            for t in keys:
                val = DT.get(t, 0.0) + DTc.get(t, 0.0)
                if (abs(np.real(val)) > eps_help) or (abs(np.imag(val)) > eps_help):
                    helpful = True
                    break

        return DT, DTc, helpful

    if HAVE_JOBLIB and len(canon_terms) > 1:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        if verbose:
            print(f"[BLISS] [screening] Using joblib with n_jobs = {n_jobs}")
        results = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(_compute_DT_DTc_and_helpful)(term, self_flag)
            for term, self_flag in zip(canon_terms, is_self_adj)
        )
        for DT, DTc, helpful in results:
            cached_DT.append(DT)
            cached_DTc.append(DTc)
            helpful_mask.append(helpful)
    else:
        if verbose and not HAVE_JOBLIB:
            print("[BLISS] [screening] joblib not available; falling back to serial T·Ne_diff builds.")
        for term, self_flag in zip(canon_terms, is_self_adj):
            DT, DTc, helpful = _compute_DT_DTc_and_helpful(term, self_flag)
            cached_DT.append(DT)
            cached_DTc.append(DTc)
            helpful_mask.append(helpful)

    # Filter to only helpful terms
    canon_terms = [t for t, keep in zip(canon_terms, helpful_mask) if keep]
    is_self_adj = [f for f, keep in zip(is_self_adj, helpful_mask) if keep]
    cached_DT   = [D for D, keep in zip(cached_DT, helpful_mask) if keep]
    cached_DTc  = [D for D, keep in zip(cached_DTc, helpful_mask) if keep]

    index_map, idx = [], 0
    for flag in is_self_adj:
        if flag:
            index_map.append((idx, None))
            idx += 1
        else:
            index_map.append((idx, idx + 1))
            idx += 2
    param_count = idx
    if verbose:
        print(f"[BLISS] Screened O-basis: {len(canon_terms)} reps kept, "
              f"parameter vector length = {param_count}")

    if param_count == 0:
        if verbose:
            print("[BLISS] No helpful O-terms found after screening. Skipping LP and returning original H.")
        return normal_ordered(H)

    # Build LP model
    if verbose:
        print("\n[BLISS] Building linear programme (with cached columns, no-new-terms constraint)…")

    # Row index over union of all terms in H, DT, DTc
    all_terms = set(H.terms.keys())
    for DT, DTc in zip(cached_DT, cached_DTc):
        all_terms.update(DT.keys())
        all_terms.update(DTc.keys())
    all_terms_sorted = sorted(all_terms)
    term2idx = {t: i for i, t in enumerate(all_terms_sorted)}
    n_terms  = len(term2idx)

    # Dense vectors for H (full H — tail only affects screening)
    H_r = np.zeros(n_terms)
    H_i = np.zeros(n_terms)
    for t, c in H.terms.items():
        k = term2idx[t]
        H_r[k] = c.real
        H_i[k] = c.imag

    # Build D_r, D_i as sparse matrices using cached DT / DTc
    rows_r, cols_r, data_r = [], [], []
    rows_i, cols_i, data_i = [], [], []

    for (self_flag, (j_re, j_im), DT, DTc) in zip(is_self_adj, index_map, cached_DT, cached_DTc):
        if self_flag:
            # single α column: O = α T  ⇒  column = DT
            for t, c in DT.items():
                k = term2idx[t]
                cr, ci = c.real, c.imag
                if cr != 0.0:
                    rows_r.append(k); cols_r.append(j_re); data_r.append(cr)
                if ci != 0.0:
                    rows_i.append(k); cols_i.append(j_re); data_i.append(ci)
        else:
            # two columns from the same cached pieces:
            # α column: (T + T†)·Ne_diff = DT + DTc
            # β column: i(T − T†)·Ne_diff = i*DT − i*DTc
            keys = set(DT.keys()) | set(DTc.keys())
            for t in keys:
                cT  = DT.get(t, 0.0)
                cTc = DTc.get(t, 0.0)
                c_alpha = cT + cTc
                c_beta  = 1j * (cT - cTc)
                k = term2idx[t]

                ar, ai = np.real(c_alpha), np.imag(c_alpha)
                br, bi = np.real(c_beta),  np.imag(c_beta)

                if ar != 0.0:
                    rows_r.append(k); cols_r.append(j_re); data_r.append(ar)
                if ai != 0.0:
                    rows_i.append(k); cols_i.append(j_re); data_i.append(ai)
                if br != 0.0:
                    rows_r.append(k); cols_r.append(j_im); data_r.append(br)
                if bi != 0.0:
                    rows_i.append(k); cols_i.append(j_im); data_i.append(bi)

    Dr_full = sp.coo_matrix(
        (data_r, (rows_r, cols_r)), shape=(n_terms, param_count)
    ).tocsc()
    Di_full = sp.coo_matrix(
        (data_i, (rows_i, cols_i)), shape=(n_terms, param_count)
    ).tocsc()

    # Prune trivial rows (no H and no D)
    row_nnz = (np.array(Dr_full.getnnz(axis=1)).ravel()
               + np.array(Di_full.getnnz(axis=1)).ravel())
    row_has_H = (H_r != 0.0) | (H_i != 0.0)
    keep_rows_mask = row_has_H | (row_nnz > 0)

    if keep_rows_mask.sum() < n_terms:
        dropped = n_terms - int(keep_rows_mask.sum())
        if verbose:
            print(f"[BLISS] [prune] Dropping {dropped} trivial rows with no H and no D contributions.")
        Dr_full = Dr_full[keep_rows_mask, :]
        Di_full = Di_full[keep_rows_mask, :]
        H_r = H_r[keep_rows_mask]
        H_i = H_i[keep_rows_mask]
        all_terms_sorted = [t for i, t in enumerate(all_terms_sorted) if keep_rows_mask[i]]
        term2idx = {t: i for i, t in enumerate(all_terms_sorted)}
        n_terms = len(all_terms_sorted)
    else:
        if verbose:
            print("[BLISS] [prune] No trivial rows to drop.")

    # No-new-terms indices (rows corresponding to terms not in original support)
    original_support = set(H.terms.keys())
    new_term_indices = np.array(
        [idx for t, idx in term2idx.items() if t not in original_support],
        dtype=int,
    )
    if verbose:
        print(
            f"[BLISS] Active rows (after pruning) = {n_terms}, "
            f"original-support rows = {len(original_support)}, "
            f"no-new-terms constraint rows = {new_term_indices.size}"
        )

    # Prune zero columns (parameters with no effect)
    col_nnz = (np.array(Dr_full.getnnz(axis=0)).ravel()
               + np.array(Di_full.getnnz(axis=0)).ravel())
    keep_cols_mask = col_nnz > 0
    if keep_cols_mask.sum() < param_count:
        dropped_cols = param_count - int(keep_cols_mask.sum())
        if verbose:
            print(f"[BLISS] [prune] Dropping {dropped_cols} parameter columns with zero effect.")
    else:
        if verbose:
            print("[BLISS] [prune] No zero columns to drop.")

    keep_cols = np.where(keep_cols_mask)[0]
    Dr = Dr_full[:, keep_cols]
    Di = Di_full[:, keep_cols]
    P_reduced = Dr.shape[1]

    # Solve LP
    if P_reduced == 0:
        if verbose:
            print("[BLISS] [warning] All columns dropped after pruning; using θ = 0.")
        theta_reduced = np.zeros(0)
    else:
        theta_reduced = solve_lp_l1_sparse(Dr, Di, H_r, H_i, new_term_indices)

    theta_opt = np.zeros(param_count)
    if P_reduced > 0:
        theta_opt[keep_cols] = theta_reduced

    # Residuals consistent with θ we actually use
    residual_r = H_r - np.asarray(Dr_full @ theta_opt).ravel()
    residual_i = H_i - np.asarray(Di_full @ theta_opt).ravel()

    # Build H_shifted directly from residual (no-new-terms, no new support)
    term2idx_residual = {t: i for i, t in enumerate(all_terms_sorted)}
    H_shifted = build_H_from_residual(
        residual_r=residual_r,
        residual_i=residual_i,
        term2idx=term2idx_residual,
        support=original_support,
    )

    if verbose:
        print("\n[BLISS] ── Optimisation complete ──")
        print(f"[BLISS] H_shift : terms = {len(H_shifted.terms):6d}, "
              f"1-norm = {sum(abs(c) for c in H_shifted.terms.values()):.6e}")
        # Sanity: term count cannot increase
        assert len(H_shifted.terms) <= len(H.terms), \
            "BUG: H_shifted has more terms than original H; this should not happen."

    return H_shifted

# ────────── helpers for pretty printing of job counts ──────────

def _fmt_jobs(label: str, n_jobs: int) -> str:
    if n_jobs in (None, -1):
        return f"{label}: {os.cpu_count() or 1}"
    return f"{label}: {n_jobs}"

# ────────── MAIN ──────────
if __name__ == "__main__":
    # Pin BLAS in main
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    if not HAM_PKL.exists():
        sys.exit(f"{HAM_PKL} not found.")

    with HAM_PKL.open("rb") as f:
        H = normal_ordered(pickle.load(f))
    n_qb = count_qubits(H)

    if HAM_PRUNE_EPS > 0.0:
        H_terms_raw = [(t, c) for (t, c) in H.terms.items() if abs(c) > HAM_PRUNE_EPS]
        H = normal_ordered(FermionOperator(dict(H_terms_raw)))

    # ───── Initial LP-BLISS shift on the original Hamiltonian ─────
    print("\n[BLISS] Initial LP-based shift on original Hamiltonian …")
    H = bliss_shift(
        H,
        n_e_target=ELEC_NUM,
        n_orb=n_qb,
        tail_l1=BLISS_TAIL_L1,
        eps_help=BLISS_EPS_HELP,
        verbose=True,
    )
    # H is now the BLISS-compressed starting Hamiltonian for the UCC loop

    occ_HF, bit_idx, ket_HF = canonical_hf_state(n_qb, ELEC_NUM)

    base_name = OUT_PKL.stem
    out_dir   = OUT_PKL.parent
    suffix    = OUT_PKL.suffix

    print(f"Spin-orbitals: {n_qb}")
    print(f"Internal spin-orbitals excluded from generators: {sorted(INTERNAL_SPINORBS)}")
    print(f"HF occupied orbitals  : {list(occ_HF)}")
    bin_str = format(bit_idx, f"0{n_qb}b")
    print(f"HF bitstring index    : {bit_idx} (bin {bin_str})\n")

    # Truncation summary
    if TRUNCATION_N_BODY is None:
        print("Truncation: NONE (full Ḣ will be reused)\n")
    else:
        mode = "number-conserving" if TRUNCATION_NUMBER_CONSERVING else "non-conserving bound"
        print(f"Truncation: ≤ {TRUNCATION_N_BODY}-body ({mode})\n")

    for k in range(1, NumberOfUnitaries + 1):
        print(f"\n=== Unitary {k}/{NumberOfUnitaries} ===")

        H_terms = list(H.terms.items())
        H_mask  = [mask_from_term(t) for (t, _c) in H_terms]

        # Context for workers — include k in ident to force refresh each unitary
        ctx = Ctx(
            H_terms=H_terms,
            H_op=H,
            H_mask=H_mask,
            n_qb=n_qb,
            ket_sparse=ket_HF,
            bit_idx=bit_idx,
            ident=(k, len(H_terms), bit_idx),
        )
        _ensure_ctx(ctx)  # init in main as well

        # E0
        E0 = expect_on_HF(H)
        print(f"Current Hamiltonian term count: {len(H_terms)}")
        print(f"⟨Φ_HF|H|Φ_HF⟩         : {E0:.12f} Ha\n")

        # Build generator pool (threads) from H
        gen_workers = (os.cpu_count() or 1)
        gens = build_generators_threaded(H_terms, occ_HF, INTERNAL_SPINORBS, gen_workers)

        # Add HF-based singles so BLISS gauge compression cannot kill singles directions
        hf_singles = build_hf_singles_generators(n_qb, occ_HF, INTERNAL_SPINORBS)
        existing_keys = {(tag, id1, id2) for (tag, id1, id2, _A) in gens}
        added = 0
        for (tag, id1, id2, A) in hf_singles:
            key = (tag, id1, id2)
            if key not in existing_keys:
                gens.append((tag, id1, id2, A))
                existing_keys.add(key)
                added += 1

        print(f"Generators after filter (S/D): {len(gens)}  (including {added} HF singles)")
        print(_fmt_jobs("Pool build threads", gen_workers))
        print(_fmt_jobs("Winner coeff threads", COEFFS_N_JOBS))
        print(_fmt_jobs("Final transform threads", TRANSFORM_N_JOBS))
        print("")

        if not gens:
            print("No generators available after filtering; stopping.")
            break

        # g(0) scan via joblib
        with parallel_backend(JOBLIB_BACKEND, inner_max_num_threads=INNER_MAX_NUM_THREADS):
            scans = Parallel(n_jobs=SCAN_N_JOBS, batch_size="auto")(
                delayed(_scan_g0_job)(ctx, g) for g in gens
            )

        scans.sort(key=lambda x: -abs(x[4]))
        print("idx |type| indices        |      g(0)")
        print("-" * 48)
        for i, (tg, i1, i2, _, g0) in enumerate(scans):
            if tg == "S":
                idx = f"{i1}->{i2}"
            else:
                idx = f"{i1}|{i2}"
            print(f"{i:3d}| {tg} | {idx:<14} | {g0: .3e}")

        tag, id1, id2, A_best, g0_best = scans[0]
        if tag == "S":
            chosen_desc = f"single {id1}->{id2}"
        else:
            chosen_desc = f"double {id1}|{id2}"
        print(f"\nChosen by |g(0)|: {chosen_desc},  |g(0)| = {abs(g0_best):.3e}\n")

        # Winner coefficients + analytic θ*
        tag, id1, id2, A_best, θ_best, E_best, g_best = optimize_single_generator(ctx, (tag, id1, id2, A_best), E0)
        print(f"Optimised θ*: {θ_best:.6f} rad")
        print(f"E*(θ*)     : {E_best:.10f} Ha")
        print(f"g*(θ*)     : {g_best:+.3e}\n")

        # Transform across term chunks; build full Ḣ then print stats
        H_bar = transform_streaming_joblib(
            ctx, A_best, θ_best,
            early_trunc=EARLY_TRUNCATE_DURING_TRANSFORM,
            n_body=(TRUNCATION_N_BODY if EARLY_TRUNCATE_DURING_TRANSFORM else None),
            number_conserving=TRUNCATION_NUMBER_CONSERVING,
        )

        print(f"Ḣ term count (pre-truncate): {len(H_bar.terms)}")
        E_bar = expect_on_HF(H_bar)
        print(f"⟨Φ_HF|Ḣ|Φ_HF⟩ (pre-truncate): {E_bar:.12f} Ha")

        # Truncate to ≤ n-body (or not); BLISS will act on this H_trunc
        if TRUNCATION_N_BODY is None:
            H_trunc = normal_ordered(H_bar)
            E_trunc = E_bar
            print("No truncation selected → using full Ḣ as input to BLISS / next iteration.")
        else:
            H_trunc = truncate_to_n_body(H_bar, TRUNCATION_N_BODY, TRUNCATION_NUMBER_CONSERVING)
            E_trunc = expect_on_HF(H_trunc)
            label = f"≤{TRUNCATION_N_BODY}-body"
            print(f"Ḣ({label}) term count     : {len(H_trunc.terms)}")
            print(f"⟨Φ_HF|Ḣ({label})|Φ_HF⟩   : {E_trunc:.12f} Ha")

        # --- LP-BLISS shift on the (possibly truncated) Hamiltonian ---
        print("\n[BLISS] Running LP-based shift on truncated Ḣ …")
        H_bliss = bliss_shift(H_trunc,
                              n_e_target=ELEC_NUM,
                              n_orb=n_qb,
                              tail_l1=BLISS_TAIL_L1,
                              eps_help=BLISS_EPS_HELP,
                              verbose=True)
        E_bliss = expect_on_HF(H_bliss)
        print(f"[BLISS] term count after shift: {len(H_bliss.terms)}")
        print(f"[BLISS] ⟨Φ_HF|H_BLISS|Φ_HF⟩ : {E_bliss:.12f} Ha")

        # This is the Hamiltonian used for the next unitary step
        H_next = normal_ordered(H_bliss)

        if SAVE_OUT:
            out_path = out_dir / f"{base_name}-{k}{suffix}"
            with out_path.open("wb") as f:
                pickle.dump(H_next, f)
            print(f"✓ Saved transformed+BLISS Hamiltonian → {out_path}")

        # Prepare for next step
        H = H_next

    print("\nAll requested unitaries processed (initial BLISS + BLISS after each step + HF singles pool).")
