"""a5b_information.py — price the `on` problem where it actually lives.

THE PROBLEM WITH ASKING A JUDGE. A per-relation judge cannot see this defect, and no
rubric fixes that. `cup on table` in isolation is a perfectly good relation: true,
sensible, worth saying once. Its worthlessness in a graph that says `on` 61% of the
time is a property of the DISTRIBUTION, not of the claim -- the 500th `on` is empty
precisely because it is the 500th, and a judge shown one relation at a time has no way
to know which one it is holding. Measured: A5b's joint info axis rated OvSGTR's `on`
relations 1.10 against 0.85 for their own non-`on` ones, i.e. it REWARDED the modal
predicate, because `on` is easy to verify and the axis had collapsed onto truth.

THE FIX IS INFORMATION-THEORETIC AND NEEDS NO JUDGE FOR THIS HALF. The information a
predicate carries is its surprisal, -log2 p(pred). Against the PSG train marginal,
`on` is worth 1.94 bits and `riding` 8.0 -- that is a measurement, not an opinion. The
judge is still needed, but only for the half it is good at (truth: false_accept 0.20
on specific corruptions), and the two combine as

    BITS OF TRUE INFORMATION PER IMAGE = sum over relations judged TRUE of -log2 p_ref

which cannot be raised by repeating a cheap predicate (each copy is worth its 1.94
bits and nothing more) nor by emitting rare predicates at random (those are false and
the judge removes them). Rarity is only rewarded when it survives verification.

REFERENCE DISTRIBUTION, and the trap in it. p_ref is the PSG TRAIN predicate marginal,
shared by both systems, so neither is scored against its own habits -- a self-
referenced surprisal would reward a model for being merely unpredictable to itself.
The trap is open-vocabulary output: our models emit predicates outside PSG's 56, and
any smoothing scheme hands those a large surprisal essentially by fiat, which would
manufacture our win. So the HEADLINE number is restricted to the shared 56-predicate
vocabulary, where both systems are measured on identical terms and no smoothing
constant can be argued about. The out-of-vocabulary mass is reported beside it, never
folded in.

Companion judge-free diversity numbers (no verdicts needed at all): the entropy of
each model's own predicate distribution, and distinct predicates per graph.

    python benchmark/a5b_information.py \
        --run runs/judge/relation_precision_psg_top10.json \
        --ref runs/packed/psg/train/meta.json

EXPECTED MODE (--expected). A judge scored with the truth rubric records P(yes) for
every relation (`relation_precision.py --rubric truth`). Weighting each relation by
P(yes) instead of thresholding it gives the EXPECTED true bits, sum of P(yes) x bits:
no threshold to choose, and a relation the judge is unsure of earns half its bits
rather than all or none. The decomposition keeps its meaning with expectations in
place of counts: accepted/img is sum P(yes), bits/accepted the P(yes)-weighted mean.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

def load_run(path: str) -> dict:
    """Read a verdicts file, transparently handling the committed.gz copies.

    runs/benchmark/a5b/ stores the raw records gzipped (6 MB -> ~126 KB), which is the
    only reason they can live in the repo at all; analysis must not need a manual
    gunzip step to use what is committed.
    """
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)



def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if not n:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a relation_precision.py output")
    p.add_argument("--ref", default="runs/packed/psg/train/meta.json",
                   help="pack meta holding predicate_counts")
    p.add_argument("--max_rank", type=int, default=10 ** 9)
    p.add_argument("--expected", action="store_true",
                   help="weight each relation by the judge's P(yes) instead of its verdict")
    p.add_argument("--out", default="")
    a = p.parse_args()

    ref = json.load(open(a.ref))["predicate_counts"]
    tot = sum(ref.values())
    bits = {k: -math.log2(v / tot) for k, v in ref.items() if v > 0}

    recs = [r for r in load_run(a.run)["records"] if r["rank"] < a.max_rank]
    by = defaultdict(list)
    for r in recs:
        by[r["system"]].append(r)

    rows = {}
    for name, v in by.items():
        imgs = {r["row"] for r in v}
        nimg = len(imgs)
        judged = [r for r in v if not r["control"]]          # verdicts usable
        emitted = v                                          # graph size incl. controls
        true = [r for r in judged if r["true"] == "yes"]

        inv = [r for r in judged if r["claim"][1] in bits]
        # w = 1 for an accepted relation and 0 otherwise, or P(yes) in expected mode
        w = (lambda r: float(r["p_yes"])) if a.expected else (
            lambda r: 1.0 if r["true"] == "yes" else 0.0)
        # Scale to the full graph: verdict rates come from non-controls, graph size
        # from everything emitted (a control was emitted, only its verdict is unusable).
        scale = len(emitted) / max(1, len(judged))

        n_true = sum(w(r) for r in judged)
        w_inv = sum(w(r) for r in inv)
        b_true = sum(w(r) * bits[r["claim"][1]] for r in inv)
        rows[name] = {
            "n_images": nimg,
            "rel_per_image": len(emitted) / nimg,
            "precision": n_true / max(1, len(judged)),
            "in_vocab_frac": len(inv) / max(1, len(judged)),
            "bits_per_true_rel": b_true / w_inv if w_inv else 0.0,
            "true_bits_per_image": b_true * scale / nimg,
            "true_rel_per_image": n_true * scale / nimg,
            "pred_entropy": entropy(Counter(r["claim"][1] for r in judged)),
            # the most frequent predicate and its share of the graph (judge-free)
            **dict(zip(("modal_pred", "modal_share"), next(
                ((k, c / len(judged)) for k, c in
                 Counter(r["claim"][1] for r in judged).most_common(1)), (None, 0.0)))),
            "distinct_true_pred_per_graph": (
                sum(len({r["claim"][1] for r in g})
                    for g in _group(true).values()) / max(1, len(_group(true)))),
        }

    hdr = (f"{'system':<24}{'rel/img':>8}{'prec':>7}{'inVoc':>7}{'bits/true':>11}"
           f"{'TRUE BITS/IMG':>15}{'H(pred)':>9}{'distinct/gr':>12}")
    print(f"\nBits of true information per image   (source: {a.run})")
    print(f"reference: PSG train marginal, {len(bits)} predicates; "
          f"`on` = {bits.get('on', float('nan')):.2f} bits, "
          f"`riding` = {bits.get('riding', float('nan')):.2f} bits")
    print(hdr)
    print("-" * len(hdr))
    for k, s in rows.items():
        # A system most of whose predicates fall outside the reference vocabulary is
        # NOT on this scale: its bits are summed over a biased subsample (the commoner
        # predicates, which are the cheap ones), so the figure understates it by an
        # unknown amount. Suppress it rather than print a number that invites the
        # wrong comparison -- the whole point of restricting to the shared vocabulary
        # is that no constant is smuggled in.
        ok = s["in_vocab_frac"] >= 0.9
        s["bits_comparable"] = bool(ok)
        bt = f"{s['true_bits_per_image']:>15.1f}" if ok else f"{'n/c':>15}"
        bp = f"{s['bits_per_true_rel']:>11.2f}" if ok else f"{'n/c':>11}"
        print(f"{k:<24}{s['rel_per_image']:>8.1f}{s['precision']:>7.3f}"
              f"{s['in_vocab_frac']:>7.2f}{bp}{bt}{s['pred_entropy']:>9.2f}"
              f"{s['distinct_true_pred_per_graph']:>12.1f}")
    if any(not r["bits_comparable"] for r in rows.values()):
        bad = [k for k, r in rows.items() if not r["bits_comparable"]]
        print(f"\n  n/c = not comparable: {', '.join(bad)} emit most predicates "
              f"OUTSIDE the reference\n  vocabulary, so a bits total over the "
              f"in-vocabulary remainder is a biased subsample.\n  Their diversity is "
              f"still readable judge-free in H(pred) and distinct/gr.")
    print("\nTRUE BITS/IMG counts only predicates inside the shared PSG vocabulary, so "
          "no smoothing constant\nis doing any work; inVoc is the fraction that "
          "qualifies. H(pred) and distinct/gr need no judge.")

    if a.out:
        json.dump({"source": a.run, "ref": a.ref,
                   "statistic": "expected (P(yes)-weighted)" if a.expected else "thresholded",
                   "per_system": rows},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


def _group(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["row"]].append(r)
    return d


if __name__ == "__main__":
    main()
