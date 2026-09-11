# Withdrawn results

The files here are **not results** — they're kept for the record. Each file
corresponds to a release that has already been withdrawn, kept as-is so
"what was withdrawn" stays checkable, not so anyone cites the numbers in
them.

## `T6_rerank_subset_2026-09-11_WITHDRAWN.{csv,txt}`

The first version of the Boltz-2 recall-backfill rerank output, published on
2026-09-11 and withdrawn the same day.

**Why it can't be used:** this run split shards by **complex**, not by
target. Each of the four shards covers all 12 targets, so a shard crashing
didn't drop a few complete targets — it bit **a third off of every
target**: per-target completion rate had a median of 68.7%, a range of
66.4%–70.8%, and **not a single target was complete**. An AUROC computed on
two-thirds of a target's candidates isn't "the full-set AUROC with wider
error bars" — it's a different quantity; P@5 on a candidate set missing a
third of its members is even less interpretable.

**On top of that, the retrieval arm in this design is unusable on its own
terms** (what got backfilled was exactly the actives retrieval had missed —
retrieval AUROC was pinned at exactly 0 for 5 of the 12 targets). So the
`retrieval` rows in the csv **can't be cited even once coverage is backfilled
to 100%**. The only valid null hypothesis is random ranking.

Full explanation in [`tasks/T6-physics.md`](../../tasks/T6-physics.md),
section "Retraction: the 09-11 numbers were computed on incomplete candidate
sets".
