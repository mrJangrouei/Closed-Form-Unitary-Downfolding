#!/usr/bin/env python3
"""
Steepest-gradient single-generator similarity transform (iterated, fast & screened)
— Matrix-free + joblib-parallel version —

This version is tuned to reproduce the **original script's outputs** exactly
(up to floating-point rounding), while adding **general n-body truncation control**:
  • Analytic line search (no SciPy): solve g(θ)=0 via quartic in t=tan(θ/2),
    evaluate E(θ) on all real candidates (+ {0,π}) and pick the minimizer.
  • Build full Ḣ first, then **print pre-truncate stats**, then (optionally) truncate to ≤ n‑body
    and print again (no early truncation during transform unless toggled).
  • Same screening logic; deterministic ordering for reproducibility.

New in this version
-------------------
- Configure truncation order with `TRUNCATION_N_BODY` (int | None).
  * Set to an integer n (e.g., 1, 2, 3, …) to keep up to n‑body number-conserving terms.
  * Set to `None` for **no truncation** (i.e., reuse full Ḣ).
- Control number conservation via `TRUNCATION_NUMBER_CONSERVING`.
  * If True (default), keeps only terms with equal numbers of creators and annihilators, up to n.
  * If False, keeps monomials where `max(creators, annihilators) ≤ n`.
- Optional early truncation during the streaming transform via `EARLY_TRUNCATE_DURING_TRANSFORM`.

Requires: openfermion ≥ 1.5, numpy, joblib (optional: falls back to threads).
"""

import os, math, pickle, warnings, sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
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

from openfermion import (
    FermionOperator, normal_ordered, hermitian_conjugated,
    count_qubits
)

# ────────── USER SETTINGS ──────────
HAM_PKL      = Path("hamiltonian_ST.pkl")
ELEC_NUM     = 4
INTERNAL_SPINORBS = {2,3,4,5}  # exclude generators whose indices are all inside this set
TOL          = 1e-10                    # match original alpha-test tolerance
SAVE_OUT     = True
OUT_PKL      = HAM_PKL.with_stem(HAM_PKL.stem + "_minE_transformed")
NumberOfUnitaries = 200                  # successive transforms

# Optional speed toggle
HAM_PRUNE_EPS = 0.0                    # e.g., 1e-12 to drop tiny H terms; 0.0 disables

# joblib controls
JOBLIB_BACKEND        = "loky"        # "loky" or "threads"
SCAN_N_JOBS           = -1            # g(0) scan
COEFFS_N_JOBS         = -1            # winner coeffs
TRANSFORM_N_JOBS      = -1            # transform chunks
INNER_MAX_NUM_THREADS = 1             # pin BLAS in workers

# Transform / truncation behavior
EARLY_TRUNCATE_DURING_TRANSFORM = False
TRUNCATION_N_BODY: Optional[int] = None   # e.g. 1,2,3,… or None for **no truncation**
TRUNCATION_NUMBER_CONSERVING     = False

# ────────── BASIC HELPERS ──────────
l1   = lambda op: sum(abs(c) for c in normal_ordered(op).terms.values())
comm = lambda X, Y: normal_ordered(X*Y - Y*X)
dcomm= lambda X, Y: comm(comm(X, Y), Y)

def alpha(A: FermionOperator, h: FermionOperator) -> int:
    """Evangelista α test: α∈{1,4} cases supported."""
    S = normal_ordered(A * comm(h, A) * A)
    if l1(S) <= TOL:                 return 1
    if l1(S - comm(h, A)) <= TOL:    return 4
    raise RuntimeError("α-test failed (neither α=1 nor α=4).")

# ────────── Matrix-free Fock-basis utilities ──────────
SparseState = Dict[int, complex]  # basis index → amplitude

def popcount_below(x: int, p: int) -> int:
    if p <= 0:
        return 0
    return (x & ((1 << p) - 1)).bit_count()

def apply_term_to_basis(term: Tuple[Tuple[int,int], ...], bitidx: int, amp: complex = 1.0) -> Tuple[int, complex] | None:
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

def g_theta(th, e11,e21,e14,e24):
    sinθ, cosθ = math.sin(th), math.cos(th)
    sin2, cos2 = math.sin(2*th), math.cos(2*th)
    return cosθ*e11 + sinθ*e21 + cos2*e14 + 0.5*sin2*e24

def E_theta(th, e00,e11,e21,e14,e24):
    sinθ, cosθ = math.sin(th), math.cos(th)
    sin2 = math.sin(2*th)
    return (e00 + sinθ*e11 + (1-cosθ)*e21 + 0.5*sin2*e14 + 0.5*(sinθ**2)*e24)

# ────────── Globals (lazy-initialized per worker) ──────────
GLOB = {}
# Keys: H_terms, H_op, H_mask, n_qb, ket_sparse, bit_idx, Hket_sparse, _ident

@dataclass(frozen=True)
class Ctx:
    H_terms: List[Tuple[Tuple[Tuple[int,int],...], complex]]
    H_op: FermionOperator
    H_mask: List[int]
    n_qb: int
    ket_sparse: SparseState
    bit_idx: int
    ident: Tuple[int,int,int]  # (epoch_k, len(H_terms), bit_idx)

def _init_worker(H_terms, H_op, H_mask, n_qb, ket_sparse, bit_idx):
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
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

def _gen_from_term(term, occ_set, internal_set):
    inds = {p for (p, _a) in term}
    if inds and inds.issubset(internal_set):
        return None
    T = FermionOperator(term, 1.0)
    X = hermitian_conjugated(T)
    if not op_acts_nonzero_on_HF(X, occ_set):
        return None
    ladd = next(iter(X.terms.keys()))
    creators = tuple(p for (p,a) in ladd if a == 1)
    annihils = tuple(p for (p,a) in ladd if a == 0)
    if   len(creators)==1 and len(annihils)==1:
        tag,id1,id2 = "S", creators[0], annihils[0]
    elif len(creators)==2 and len(annihils)==2:
        tag,id1,id2 = "D", tuple(creators), tuple(annihils)
    else:
        return None
    A = normal_ordered(X - hermitian_conjugated(X))
    if not A.terms:
        return None
    key = tuple(sorted(A.terms.items()))
    return (tag, id1, id2, A, key)

def build_generators_threaded(H_terms, occ_HF, internal_set, max_workers):
    occ_set = set(occ_HF)
    with ThreadPoolExecutor(max_workers=max_workers) as tpool:
        cands = tpool.map(lambda tc: _gen_from_term(tc[0], occ_set, internal_set), H_terms)
    unique = {}
    for item in cands:
        if item is None:
            continue
        tag, id1, id2, A, key = item
        if key not in unique:
            unique[key] = (tag, id1, id2, A)
    return list(unique.values())

# ────────── g(0) scan (joblib)

def _scan_g0_job(ctx: Ctx, gen):
    _ensure_ctx(ctx)
    tag, id1, id2, A = gen
    ket  = GLOB["ket_sparse"]
    Hket = GLOB["Hket_sparse"]
    psi  = apply_op_to_sparse(A, ket)
    g0   = 2.0 * float(np.real(vdot_sparse(Hket, psi)))
    return tag, id1, id2, A, g0

# ────────── Analytic θ* via quartic in t = tan(θ/2)

def _theta_star_via_roots(e00, e11, e21, e14, e24):
    coeffs = np.array([e14 - e11, 2.0*(e21 - e24), -6.0*e14, 2.0*(e21 + e24), e11 + e14], dtype=float)
    nz = np.flatnonzero(np.abs(coeffs) > 1e-14)
    if nz.size == 0:
        ths = [0.0, math.pi]
    else:
        coeffs = coeffs[nz[0]:]
        roots = np.roots(coeffs)
        ts = [r.real for r in roots if abs(r.imag) < 1e-10]
        ths = [0.0, math.pi] + [2.0*math.atan(t) for t in ts]
    norm = lambda th: (th + 2*math.pi) % (2*math.pi)
    cand = sorted({round(norm(th), 12) for th in ths})
    best = None
    for th in cand:
        val = E_theta(th, e00, e11, e21, e14, e24)
        if (best is None) or (val < best[1]):
            best = (th, val)
    th_star, E_star = best
    g_star = g_theta(th_star, e11, e21, e14, e24)
    th_star = (th_star + 2*math.pi) % (2*math.pi)
    return th_star, float(E_star), float(g_star)

# ────────── Winner coeffs (joblib across slices)

def _coeffs_chunk_job(ctx: Ctx, idx_slice: List[int], A: FermionOperator, ket: SparseState, psi: SparseState, psi2: SparseState, A_mask: int):
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]; H_mask = GLOB["H_mask"]
    acc = {1: {'E1':0.0,'E2':0.0}, 4: {'E1':0.0,'E2':0.0}}
    for i in idx_slice:
        if (H_mask[i] & A_mask) == 0:
            continue
        term, ch = H_terms[i]
        h = FermionOperator(term, ch)
        a = alpha(A, h)
        phi  = apply_op_to_sparse(h,  ket)
        hpsi = apply_op_to_sparse(h,  psi)
        E1 = vdot_sparse(ket, hpsi) + vdot_sparse(psi,  phi)
        E2 = vdot_sparse(ket, apply_op_to_sparse(h, psi2)) + 2.0*vdot_sparse(psi, hpsi) + vdot_sparse(psi2, phi)
        acc[a]['E1'] += float(np.real(E1))
        acc[a]['E2'] += float(np.real(E2))
    return acc

def optimize_single_generator(ctx: Ctx, gen, e00):
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]; nT = len(H_terms)
    ket     = GLOB["ket_sparse"]
    tag, id1, id2, A = gen

    psi  = apply_op_to_sparse(A, ket)
    psi2 = apply_op_to_sparse(A, psi)
    A_mask = mask_from_A(A)

    chunk = max(1, nT//( (os.cpu_count() or 1) * 4 ))
    idxs  = list(range(nT))
    slices = [idxs[i:i+chunk] for i in range(0, nT, chunk)]

    coeff = {1: {'E1': 0.0, 'E2': 0.0}, 4: {'E1': 0.0, 'E2': 0.0}}
    with parallel_backend(JOBLIB_BACKEND, inner_max_num_threads=INNER_MAX_NUM_THREADS):
        parts = Parallel(n_jobs=COEFFS_N_JOBS, batch_size="auto")(
            delayed(_coeffs_chunk_job)(ctx, sl, A, ket, psi, psi2, A_mask) for sl in slices
        )
    for acc in parts:
        for k in (1,4):
            coeff[k]['E1'] += acc[k]['E1']
            coeff[k]['E2'] += acc[k]['E2']

    e11, e21 = coeff[1]['E1'], coeff[1]['E2']
    e14, e24 = coeff[4]['E1'], coeff[4]['E2']

    θ, E, g = _theta_star_via_roots(e00, e11, e21, e14, e24)
    θ = (θ + 2*math.pi) % (2*math.pi)
    return tag, id1, id2, A, θ, E, g

# ────────── General ≤ n‑body truncation ──────────

def _term_counts(term) -> Tuple[int,int]:
    creators = sum(1 for (_p,a) in term if a == 1)
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

def truncate_to_n_body(H_op: FermionOperator, n_body: Optional[int], number_conserving: bool = True) -> FermionOperator:
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
        return O + math.sin(θ)*OA + (1-math.cos(θ))*OAA
    return O + 0.5*math.sin(2*θ)*OA + 0.5*(math.sin(θ)**2)*OAA

def _transform_chunk_job(ctx: Ctx, idx_slice: List[int], A: FermionOperator, θ: float, A_mask: int,
                          early_trunc: bool, n_body: Optional[int], number_conserving: bool) -> FermionOperator:
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]; H_mask = GLOB["H_mask"]
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

def transform_streaming_joblib(ctx: Ctx, A: FermionOperator, θ: float, early_trunc: bool,
                               n_body: Optional[int], number_conserving: bool) -> FermionOperator:
    _ensure_ctx(ctx)
    H_terms = GLOB["H_terms"]; nT = len(H_terms)
    A_mask = mask_from_A(A)
    chunk = max(1, nT//( (os.cpu_count() or 1) * 4 ))
    idxs  = list(range(nT))
    slices = [idxs[i:i+chunk] for i in range(0, nT, chunk)]

    with parallel_backend(JOBLIB_BACKEND, inner_max_num_threads=INNER_MAX_NUM_THREADS):
        parts = Parallel(n_jobs=TRANSFORM_N_JOBS, batch_size="auto")(
            delayed(_transform_chunk_job)(ctx, sl, A, θ, A_mask, early_trunc, n_body, number_conserving) for sl in slices
        )

    # Pairwise reduce to keep peak small
    while len(parts) > 1:
        nxt: List[FermionOperator] = []
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                nxt.append(normal_ordered(parts[i] + parts[i+1]))
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

# ────────── helpers for pretty printing of job counts ──────────

def _fmt_jobs(label: str, n_jobs: int) -> str:
    if n_jobs in (None, -1):
        return f"{label}: {os.cpu_count() or 1}"
    return f"{label}: {n_jobs}"

# ────────── MAIN ──────────
if __name__=="__main__":
    # Pin BLAS in main
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    if not HAM_PKL.exists():
        sys.exit(f"{HAM_PKL} not found.")

    with HAM_PKL.open("rb") as f:
        H = normal_ordered(pickle.load(f))
    n_qb = count_qubits(H)

    if HAM_PRUNE_EPS > 0.0:
        H_terms_raw = [(t,c) for (t,c) in H.terms.items() if abs(c) > HAM_PRUNE_EPS]
        H = normal_ordered(FermionOperator(dict(H_terms_raw)))

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

        # Context for workers — **minimal fix**: include k in ident to force refresh each unitary
        ctx = Ctx(
            H_terms=H_terms, H_op=H, H_mask=H_mask, n_qb=n_qb, ket_sparse=ket_HF, bit_idx=bit_idx,
            ident=(k, len(H_terms), bit_idx)  # ← epoch included
        )
        _ensure_ctx(ctx)  # init in main as well

        # E0
        E0 = expect_on_HF(H)
        print(f"Current Hamiltonian term count: {len(H_terms)}")
        print(f"⟨Φ_HF|H|Φ_HF⟩         : {E0:.12f} Ha\n")

        # Build generator pool (threads)
        gen_workers = (os.cpu_count() or 1)
        gens = build_generators_threaded(H_terms, occ_HF, INTERNAL_SPINORBS, gen_workers)
        print(f"Generators after filter (S/D): {len(gens)}")
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
        print("-"*48)
        for i,(tg,i1,i2,_,g0) in enumerate(scans):
            idx = f"{i1}->{i2}" if tg=="S" else f"{i1}|{i2}"
            print(f"{i:3d}| {tg} | {idx:<14} | {g0: .3e}")

        tag,id1,id2,A_best,g0_best = scans[0]
        chosen_desc = f"single {id1}->{id2}" if tag=="S" else f"double {id1}|{id2}"
        print(f"\nChosen by |g(0)|: {chosen_desc},  |g(0)| = {abs(g0_best):.3e}\n")

        # Winner coefficients + analytic θ*
        tag,id1,id2,A_best,θ_best,E_best,g_best = optimize_single_generator(ctx, (tag,id1,id2,A_best), E0)
        print(f"Optimised θ*: {θ_best:.6f} rad")
        print(f"E*(θ*)     : {E_best:.10f} Ha")
        print(f"g*(θ*)     : {g_best:+.3e}\n")

        # Transform across term chunks; build full Ḣ then print original-style stats
        H_bar = transform_streaming_joblib(
            ctx, A_best, θ_best,
            early_trunc=EARLY_TRUNCATE_DURING_TRANSFORM,
            n_body=(TRUNCATION_N_BODY if EARLY_TRUNCATE_DURING_TRANSFORM else None),
            number_conserving=TRUNCATION_NUMBER_CONSERVING,
        )

        print(f"Ḣ term count (pre-truncate): {len(H_bar.terms)}")
        E_bar = expect_on_HF(H_bar)
        print(f"⟨Φ_HF|Ḣ|Φ_HF⟩ (pre-truncate): {E_bar:.12f} Ha")

        # Truncate to ≤ n‑body (or not) and save; reuse for next iteration
        if TRUNCATION_N_BODY is None:
            H_next = normal_ordered(H_bar)
            E_next = E_bar
            print("No truncation selected → reusing full Ḣ for next iteration.")
        else:
            H_next = truncate_to_n_body(H_bar, TRUNCATION_N_BODY, TRUNCATION_NUMBER_CONSERVING)
            E_next = expect_on_HF(H_next)
            label = f"≤{TRUNCATION_N_BODY}-body"
            print(f"Ḣ({label}) term count     : {len(H_next.terms)}")
            print(f"⟨Φ_HF|Ḣ({label})|Φ_HF⟩   : {E_next:.12f} Ha")

        if SAVE_OUT:
            out_path = out_dir / f"{base_name}-{k}{suffix}"
            with out_path.open("wb") as f:
                pickle.dump(H_next, f)
            print(f"✓ Saved transformed Hamiltonian → {out_path}")

        # Prepare for next step
        H = normal_ordered(H_next)

    print("\nAll requested unitaries processed.")

