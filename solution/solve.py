#!/usr/bin/env python3
"""Oracle reference solution (standard library only).

Recovers the retired data-sharding tool's assignment-signature routine from /app/data/archive.jsonl and
appends the compatibility extension to PyTorch's /app/torch/utils/data/distributed.py. Nothing about the
routine is assumed that the archive does not pin --
not its constants and not its STRUCTURE.

Stage 1 (the sharding gate) is documented in instruction.md: apply the documented shuffle, tail and mode
rules to reduce each query to its sequence of bounded rank codes.

Stage 2 (the signature over that code sequence) is undocumented. It is a modular-polynomial fold
    acc = 0; for code in codes: acc = (acc * P + code) mod M
followed by a bijective, data-dependent bit scramble on the 20-bit accumulator and a rotation keyed on the
queried-index count. This program DERIVES both the scramble's structure and every constant from the
archive by searching a family of the form

    round 1:  v ^= (v << (1 + (v & (2**S1 - 1)))) & mask1          # shift read from the low S1 bits
    round 2:  v ^= (TABLE[(v >> IO) & (2**IW - 1)] << (IO+IW)) & mask2   # 2**IW-entry substitution
    round 3:  v ^= (v >> (1 + ((v >> (WORD-S3)) & (2**S3 - 1)))) & mask3 # shift read from the top S3 bits
    signature = rotl(v, ROTS[count mod 5])                        # class 0 unrotated

over the structural parameters (WORD, S1, IW, IO, S3), keeping the single configuration whose constants
reproduce the WHOLE archive. Every round writes only bits its own selector does not read, so every round
inverts and its shift/index stays readable from its output -- which is what lets the archive determine the
routine. WORD is read off the widest signature. A single-index query reduces to one rank code, so its
accumulator is that code (a known input) and the scramble is observed there, seeding some table entries and
the one-index rotation. Two-index runs whose scramble index those entries cover give accumulators
(c0*P + c1) mod M with the codes known; eliminating P between two of them leaves a multiple of M, and the
gcd of enough of them is M, after which one modular inverse gives P. With M and P known every archived
accumulator is known, so the whole table and the remaining rotations drop out, and the recovered routine is
checked against the entire archive.
"""
import json
import sys
from math import gcd
from pathlib import Path

ARCHIVE = Path("/app/data/archive.jsonl")
OUT = Path("/app/torch/utils/data/distributed.py")
TEMPLATE = Path(__file__).resolve().parent / "sharding_signature_extension_template.py"

# --- the DOCUMENTED sharding vocabulary (public; stated in instruction.md) -----------------------
MAX_W = 8
RANK_CODE = [101, 134, 167, 200, 233, 266, 299, 332]
DROPPED_CODE = 68
LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
MASK64 = (1 << 64) - 1


# --- documented stage-1 sharding gate ------------------------------------------------------------
def shuffle_order(N, seed):
    order = list(range(N))
    if seed is None:
        return order
    state = seed & MASK64
    for i in range(N - 1, 0, -1):
        state = (state * LCG_A + LCG_C) & MASK64
        j = (state >> 33) % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def code_of_index(query):
    N, W = query["N"], query["W"]
    mode, tail, seed = query["mode"], query["tail"], query["seed"]
    order = shuffle_order(N, seed)
    pos = [0] * N
    for p, g in enumerate(order):
        pos[g] = p
    if tail == "drop_last":
        keep = (N // W) * W
        assign_len = keep
    else:
        assign_len = ((N + W - 1) // W) * W
    per_rank = assign_len // W if W else 0
    out = [None] * N
    for g in range(N):
        p = pos[g]
        if tail == "drop_last" and p >= keep:
            out[g] = DROPPED_CODE
        elif mode == "contiguous":
            out[g] = RANK_CODE[p // per_rank]
        else:
            out[g] = RANK_CODE[p % W]
    return out


def code_seq(query):
    code = code_of_index(query)
    return [code[i] for i in query["indices"]]


# --- the scramble family, as structure with parameters and constants left open -------------------
class Scramble:
    """The 3-round data-dependent bijection, parameterised by structural widths. Given the structural
    parameters, round shifts and the table index are computed from the value's own bits; only TABLE stays
    a set of unknown constants."""

    def __init__(self, word, s1, iw, io, s3, table=None):
        self.WORD = word
        self.WMASK = (1 << word) - 1
        self.S1, self.IW, self.IO, self.S3 = s1, iw, io, s3
        self.R1MASK = self.WMASK ^ ((1 << s1) - 1)
        self.R2MASK = self.WMASK ^ ((1 << (io + iw)) - 1)
        self.R3MASK = (1 << (word - s3)) - 1
        self.LOWMASK = (1 << (io + iw)) - 1
        self.ENTRY = (1 << (word - io - iw)) - 1
        self.IMASK = (1 << iw) - 1
        self.table = table

    def round1(self, v):
        return (v ^ ((v << (1 + (v & ((1 << self.S1) - 1)))) & self.R1MASK)) & self.WMASK

    def round1_inv(self, w):
        shift = 1 + (w & ((1 << self.S1) - 1))
        v = w & ((1 << self.S1) - 1)
        for i in range(self.S1, self.WORD):
            t = (v >> (i - shift)) & 1 if i - shift >= 0 else 0
            v |= (((w >> i) & 1) ^ t) << i
        return v

    def round3(self, v):
        return (v ^ ((v >> (1 + ((v >> (self.WORD - self.S3)) & ((1 << self.S3) - 1)))) & self.R3MASK)) \
            & self.WMASK

    def round3_inv(self, w):
        shift = 1 + ((w >> (self.WORD - self.S3)) & ((1 << self.S3) - 1))
        v = w & (((1 << self.S3) - 1) << (self.WORD - self.S3))
        for i in range(self.WORD - self.S3 - 1, -1, -1):
            t = (v >> (i + shift)) & 1 if i + shift < self.WORD else 0
            v |= (((w >> i) & 1) ^ t) << i
        return v

    def round2(self, v):
        idx = (v >> self.IO) & self.IMASK
        return (v ^ ((self.table[idx] << (self.IO + self.IW)) & self.R2MASK)) & self.WMASK

    def rotl(self, v, r):
        r %= self.WORD
        return ((v << r) | (v >> (self.WORD - r))) & self.WMASK

    def rotr(self, v, r):
        return self.rotl(v, (self.WORD - (r % self.WORD)) % self.WORD)

    def index_of(self, acc):
        return (self.round1(acc & self.WMASK) >> self.IO) & self.IMASK

    def encode(self, acc, count, rots):
        v = self.round3(self.round2(self.round1(acc & self.WMASK)))
        return self.rotl(v, rots[count % 5])


class SolveError(Exception):
    pass


def _prime_factors(n):
    n = abs(n)
    fs, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def _fold(seq, M, P):
    acc = 0
    for c in seq:
        acc = (acc * P + c) % M
    return acc


def _attempt(rows, sc, max_code):
    """Given a fixed structural family `sc` (no table yet), try to recover ROTS, M, P and TABLE and
    verify the whole archive. Returns the full recovery dict or raises SolveError."""
    WORD = sc.WORD
    probes = [r for r in rows if r["n"] == 1]
    twos = [r for r in rows if r["n"] == 2]
    if len(probes) < 5 or len(twos) < 8:
        raise SolveError("too few single/two-index runs")

    # step 1: single-index probes have a known accumulator (the code), so for each candidate one-index
    # rotation the scramble can be run forward to bits IO.. and inverted backward from the signature; a
    # rotation is consistent iff the low bits agree and every table index the probes reach reads the same
    # entry. That gives the one-index rotation and a partial table.
    rot1_cands = []
    for r1 in range(WORD):
        partial, ok = {}, True
        for r in probes:
            code = r["seq"][0]
            v1 = sc.round1(code)
            idx = (v1 >> sc.IO) & sc.IMASK
            w = sc.round3_inv(sc.rotr(r["sig"], r1))
            if (w & sc.LOWMASK) != (v1 & sc.LOWMASK):
                ok = False
                break
            entry = ((w ^ v1) >> (sc.IO + sc.IW)) & sc.ENTRY
            if partial.setdefault(idx, entry) != entry:
                ok = False
                break
        if ok:
            rot1_cands.append((r1, partial))
    if not rot1_cands:
        raise SolveError("no consistent one-index rotation for this structure")

    # step 2: pin M and P from two-index runs whose scramble index lands on a seeded table entry
    for r1, partial in rot1_cands:
        for r2 in range(WORD):
            exact = []
            for r in twos:
                w = sc.round3_inv(sc.rotr(r["sig"], r2))
                idx = (w >> sc.IO) & sc.IMASK
                if idx not in partial:
                    continue
                v1 = (w ^ ((partial[idx] << (sc.IO + sc.IW)) & sc.R2MASK)) & sc.WMASK
                if (v1 & sc.LOWMASK) != (w & sc.LOWMASK):
                    continue
                acc = sc.round1_inv(v1)
                exact.append((r["seq"][0], r["seq"][1], acc))
            if len(exact) < 2:
                continue
            multiple = 0
            for i in range(len(exact)):
                c0i, c1i, ai = exact[i]
                for j in range(i + 1, len(exact)):
                    c0j, c1j, aj = exact[j]
                    multiple = gcd(multiple, abs(c0j * (ai - c1i) - c0i * (aj - c1j)))
            if multiple == 0:
                continue
            for m in _prime_factors(multiple):
                if not (max_code < m < (1 << WORD)):
                    continue
                c0, c1, a = exact[0]
                if c0 % m == 0 or a >= m:
                    continue
                try:
                    p = ((a - c1) * pow(c0 % m, -1, m)) % m
                except ValueError:
                    continue
                if all((c0k * p + c1k) % m == ak for c0k, c1k, ak in exact):
                    try:
                        return _complete(rows, sc, r1, r2, m, p)
                    except SolveError:
                        continue
    raise SolveError("no admissible modulus reproduces the two-index runs for this structure")


def _complete(rows, sc, r1, r2, M, P):
    WORD = sc.WORD
    # step 3: with M and P known, fill the whole table from rows whose rotation is known -- classes 0
    # (documented unrotated), 1 and 2 (recovered above)
    rot_of = {0: 0, 1: r1, 2: r2}
    table = {}
    for r in rows:
        rot = rot_of.get(r["n"] % 5)
        if rot is None:
            continue
        acc = _fold(r["seq"], M, P)
        v1 = sc.round1(acc)
        idx = (v1 >> sc.IO) & sc.IMASK
        w = sc.round3_inv(sc.rotr(r["sig"], rot))
        if (w & sc.LOWMASK) != (v1 & sc.LOWMASK):
            raise SolveError("a known-rotation row is inconsistent with this candidate")
        entry = ((w ^ v1) >> (sc.IO + sc.IW)) & sc.ENTRY
        if table.setdefault(idx, entry) != entry:
            raise SolveError("the table is inconsistent for this candidate")
    if len(table) != (1 << sc.IW):
        raise SolveError(f"the archive pins only {len(table)}/{1 << sc.IW} table entries")
    sc.table = [table[i] for i in range(1 << sc.IW)]

    # step 4: the remaining rotations (classes 3, 4), then a full-archive verification
    rots = [0, r1, r2, None, None]
    for cls in (3, 4):
        members = [r for r in rows if r["n"] % 5 == cls]
        if not members:
            raise SolveError(f"the archive never exercises queried-index-count class {cls}")
        found = [rot for rot in range(WORD)
                 if all(sc.encode(_fold(r["seq"], M, P), r["n"], [0, r1, r2, rot, rot]) == r["sig"]
                        for r in members if r["n"] % 5 == cls)]
        if len(found) != 1:
            raise SolveError(f"class {cls} does not pin its rotation: {found}")
        rots[cls] = found[0]

    for r in rows:
        if sc.encode(_fold(r["seq"], M, P), r["n"], rots) != r["sig"]:
            raise SolveError("the recovered routine does not reproduce the archive")

    return {"WORD": WORD, "S1": sc.S1, "IW": sc.IW, "IO": sc.IO, "S3": sc.S3,
            "M": M, "P": P, "TABLE": list(sc.table), "ROTS": rots}


def recover(archive):
    """Derive the structure and constants of the signature routine from an archive of (query, signature)
    rows. Returns a dict with WORD, S1, IW, IO, S3, M, P, TABLE, ROTS, or raises SolveError. The structural
    family is searched and only the single configuration that reproduces the whole archive is kept."""
    rows = []
    for query, sig in archive:
        seq = code_seq(query)
        rows.append({"n": len(seq), "seq": seq, "sig": sig})
    if not rows:
        raise SolveError("the archive is empty")
    max_code = max((max(r["seq"]) for r in rows if r["seq"]), default=0)
    WORD = max(r["sig"].bit_length() for r in rows)          # read off the widest signature

    solutions = []
    for s1 in (2, 3, 4):
        for iw in (4, 5, 6):
            for io in (2, 3, 4):
                for s3 in (2, 3, 4):
                    if io + iw >= WORD - 1 or io + iw + 1 > WORD - s3:
                        continue                              # no room for a table write / round 3
                    sc = Scramble(WORD, s1, iw, io, s3)
                    try:
                        rec = _attempt(rows, sc, max_code)
                    except SolveError:
                        continue
                    solutions.append(rec)
    if not solutions:
        raise SolveError("no structural family reproduced the archive")
    if len(solutions) > 1:
        raise SolveError(f"the structure is not uniquely determined ({len(solutions)} families fit)")
    return solutions[0]


def apply(rec, query):
    """Signature of a query under recovered structure + constants -- used by the fixture generator to prove
    the archive determines the routine over the held-out domain."""
    sc = Scramble(rec["WORD"], rec["S1"], rec["IW"], rec["IO"], rec["S3"], rec["TABLE"])
    seq = code_seq(query)
    return sc.encode(_fold(seq, rec["M"], rec["P"]), len(seq), rec["ROTS"])


def main():
    archive = []
    for line in ARCHIVE.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            archive.append((rec["query"], rec["signature"]))
    try:
        rec = recover(archive)
    except SolveError as exc:
        print("solve.py: " + str(exc), file=sys.stderr)
        raise SystemExit(1)

    src = TEMPLATE.read_text()
    for key in ("WORD", "S1", "IW", "IO", "S3", "M", "P", "TABLE", "ROTS"):
        src = src.replace(f"__{key}__", str(rec[key]))
    with OUT.open("a", newline="\n") as target:
        target.write(src)

    for query, sig in archive:
        assert apply(rec, query) == sig

    print(f"recovered structure WORD={rec['WORD']} S1={rec['S1']} IW={rec['IW']} IO={rec['IO']} "
          f"S3={rec['S3']}")
    print(f"recovered M={rec['M']} P={rec['P']} rotations={rec['ROTS']}")
    print(f"reproduced all {len(archive)} archived signatures; appended extension to {OUT}")


if __name__ == "__main__":
    main()
