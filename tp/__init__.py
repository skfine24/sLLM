"""Tensor-parallel plumbing for the DGX Spark pair (TP2).

`topology` resolves the 2-node cluster layout (head + worker), `collectives`
provides the NCCL contract plus a SimCollectives harness used by tests and by
the CPU oracle, and `rank_table` maps checkpoint tensors to per-rank slices
under TP2. Model qwen4_exp/deepseek_v4 arch-specific slicing lives in
`loaders.tp_shard`; this package is the cluster-neutral part.
"""
