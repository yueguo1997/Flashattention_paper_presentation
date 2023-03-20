# Flashattention_paper_presentation


## Background

### Problem Overview

Transformers have become larger and more complex, but incorporating longer context into them is still a challenging task because their self-attention module has a time and memory complexity that scales quadratically with sequence length. The problem this paper aimed to addresss is how to enhance the speed and memory efficiency of attention in order to assist Transformer models in overcoming their memory and runtime obstacles for long sequences.

Many other methods have been developed. However, many of these methods do not show a significant improvement in speed over standard attention because they primarily concentrate on reducing the number of floating-point operations (FLOPs) and neglect the overheads from memory access (IO).


### SRAM vs HBM
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/17782d32252de3ba7d854dbdfa9eb3108829bbb9/image1.png)


### Convert the problem
Our goal is to maximize the use of fast memory: SRAM.


### Methods
IO aware with carefully accounting for the number of access to the slow and fast memory through tiling and recomputing


### Recall Standard Attention
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image2.png)


## FlashAttention
### Method
* Tiling
* Recomputation


### Algorithm


![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image3.png)





## Complexity Analysis
* Standard Attetion
* FlashatAttention


## Extension: Block-Sparse FlashatAttention


## Experiments and results

### Training time

### Model Accuracy


### Memory usage


## Limitation


## Links

[Video]()

[Notebook]()

[Paper]()

[Github repo with code]()

[Presentation reference]()



