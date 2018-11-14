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

Research log

Nov. 14
1. Divide the song-level spectrograms into chunk-level ones allows the model to learn.
2. the main problem was that the input sequences were too long (information loss in the last hidden states).
3. One interesting observation is attention mechnism may not help representation learning.
