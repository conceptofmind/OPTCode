# OPTCoder
A scalable training implementation utilizing the OPT model for code generation.

### Usage
```bash
$ git clone https://github.com/conceptofmind/OPTCode.git
$ cd OPTCode
$ colossalai run --nproc_per_node 1 train.py --use_trainer
```

### Developer Updates
Developer updates can be found on: 
- https://twitter.com/EnricoShippole
- https://www.linkedin.com/in/enrico-shippole-495521b8/

### TODO:
- [ ] Add logging with Weights and Biases
- [x] Build data loaders
- [ ] Setup ColossalAI engine
- [ ] Implement ZeRO

### Author:
- Enrico Shippole

### Additional Information:
- [OPT Research Paper](https://arxiv.org/abs/2205.01068)
- [OPT Official Github](https://github.com/facebookresearch/metaseq)

### Citations:

@misc{zhang2022opt,
      title={OPT: Open Pre-trained Transformer Language Models}, 
      author={Susan Zhang and Stephen Roller and Naman Goyal and Mikel Artetxe and Moya Chen and Shuohui Chen and Christopher Dewan and Mona Diab and Xian Li and Xi Victoria Lin and Todor Mihaylov and Myle Ott and Sam Shleifer and Kurt Shuster and Daniel Simig and Punit Singh Koura and Anjali Sridhar and Tianlu Wang and Luke Zettlemoyer},
      year={2022},
      eprint={2205.01068},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}