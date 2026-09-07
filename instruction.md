Extend PyTorch's distributed sampler module at /app/torch/utils/data/distributed.py with a callable sharding_signature(query) that returns an int. Keep the existing DistributedSampler behavior intact. The new function and every helper it calls must be self-contained in that file, use only Python's standard library, and must not rely on Torch at call time.

This is a compatibility extension for a retired distributed-training sharding tool. PyTorch's existing DistributedSampler establishes the relevant per-sample sharding model: a global ordered sample list is divided across ranks with deterministic ordering and even per-rank work. The compatibility extension must reproduce archived assignment signatures for a broader version of that model.

A query is a JSON-compatible object with this schema: {"N": integer, "W": integer, "mode": "contiguous" or "interleaved", "tail": "drop_last" or "pad", "seed": integer or null, "indices": array of integers}.

N is the number of global sample indices from 0 through N-1. W is the world size and must be from 2 through 8. indices is the ordered list of named global indices. Return the signature as a plain non-negative int.

First reduce a query to rank codes. Start with the identity ordering 0, 1, through N-1 when seed is null. Otherwise, apply Fisher-Yates shuffling. Initialize the unsigned 64-bit state to seed. For each draw, set state to (state multiplied by 6364136223846793005 plus 1442695040888963407) modulo 2 to the 64th power, then use state shifted right by 33. For i from N-1 down to 1, swap positions i and drawn value modulo i+1. position(g) is the resulting position of global index g.

For drop_last, keep is (N divided by W using integer division) multiplied by W. Positions at or beyond keep are dropped. For pad, assignable is N divided by W rounded up, then multiplied by W. No sample is dropped. per_rank is keep or assignable, divided by W. A non-dropped position p maps to rank p divided by per_rank using integer division in contiguous mode, and to rank p modulo W in interleaved mode. The rank codes for ranks 0 through 7 are 101, 134, 167, 200, 233, 266, 299, and 332. A dropped sample has code 68.

The canonical reduction is the ordered sequence of codes for the query's indices. The signature depends only on this code sequence, so queries resolving to the same sequence must return the same signature.

/app/data/archive.jsonl contains the surviving archive. Each line is a JSON object with a query object and an integer signature. The archive completely determines the undocumented transform from a rank-code sequence to its 20-bit assignment signature. Standard hashes and ordinary regression do not reproduce it. Recover the transform from the archive and implement it in the requested PyTorch module. The verifier uses unseen queries, including equivalent queries, uneven tails, both assignment modes, and shuffled orders. Every result must match exactly.
