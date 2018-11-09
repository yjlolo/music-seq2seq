# music-seq2seq
This is to build a seq2seq auto-encoder for music audio, and use the learnt representations for downstream tasks.
The current downstream task in interest is emotion recognition, using the [PMEmo Dataset](http://pmemo.hellohui.cn).
The repo will not include the dataset.

Under construction
- [x] dataset
- [x] data loader
- [x] model
    - [ ] attention
    - [ ] regressive inference 
- [x] trainer
    - [x] extra loss constraints
- [ ] classifier

This is also the first repo, intended to improve coding skills as well.
The structure and template are adapted from [Pytorch-template](https://github.com/victoresque/pytorch-template).