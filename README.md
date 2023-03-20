# Flashattention_paper_presentation


## Background

### Problem Overview

Transformers have become larger and more complex, but incorporating longer context into them is still a challenging task because their self-attention module has a time and memory complexity that scales quadratically with sequence length. The problem this paper aimed to addresss is how to enhance the speed and memory efficiency of attention in order to assist Transformer models in overcoming their memory and runtime obstacles for long sequences.

Many other methods have been developed. However, many of these methods do not show a significant improvement in speed over standard attention because they primarily concentrate on reducing the number of floating-point operations (FLOPs) and neglect the overheads from memory access (IO).



### SRAM vs HBM
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/17782d32252de3ba7d854dbdfa9eb3108829bbb9/image1.png)



### Methods
Above all, one of our gaol is to restructure the attention and let all the intermediate matrics computed on fast memory: SRAM. 
IO aware with carefully accounting for the number of access to the slow and fast memory through tiling and recomputing


### Recall Standard Attention



## FlashAttention
### Method
* Tiling
* Recomputation


### Algorithm





## Complexity Analysis
* Standard Attetion
* FlashatAttention


## Extension: Block-Sparse FlashatAttention


## Experiments and results



## Limitation


## Question 1
## Question 2




## Others

[Video]()

[Notebook]()

[Paper]()

[Github repo with code]()

[Presentation reference]()



