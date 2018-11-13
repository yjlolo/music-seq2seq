Program efficiency
- group sequences with similar duration for batching

Possible solution to bad performance
- make subsequences as input data
- only focus on partial input (e.g., specific frequency bands or instruments)

Problems
- attention mechnism (Nov. 13)
    1. sequence length being too long, making it slow
    2. nan even with grad_clip, buggy at the moment
    3. no improvement in performance, not using it for now