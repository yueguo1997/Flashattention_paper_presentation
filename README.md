# Flashattention_paper_presentation


## Background

### Problem Overview

Transformers have become larger and more complex, but incorporating longer context into them is still a challenging task because their self-attention module has a time and memory complexity that scales quadratically with sequence length. The problem this paper aimed to addresss is how to enhance the speed and memory efficiency of attention in order to assist Transformer models in overcoming their memory and runtime obstacles for long sequences.

Many other methods have been developed. However, many of these methods do not show a significant improvement in speed over standard attention because they primarily concentrate on reducing the number of floating-point operations (FLOPs) and neglect the overheads from memory access (IO).


### Hardware performance 

* SRAM vs HBM

![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/17782d32252de3ba7d854dbdfa9eb3108829bbb9/image1.png)

The on-chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size.

* Time calculation of operations

Memory-bound: The time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g.,activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).



### Convert the problem
The goal is to maximize the use of fast memory: SRAM。 In the meanwhile, the 


### Methods
IO aware with carefully accounting for the number of access to the slow and fast memory through tiling and recomputing




## FlashAttention



### Recall Standard Attention
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image2.png)



### Method

* Tiling


* Recomputation


### Algorithm


![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image3.png)





## Complexity Analysis
Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑. Standard attention (Algorithm 0) requires Θ(𝑁 𝑑 + 𝑁2) HBM accesses, while FlashAttention (Algorithm 1) requires Θ(𝑁2𝑑2𝑀−1) HBM accesses.

* Standard Attetion

* FlashatAttention



## Extension: Block-Sparse FlashatAttention
Get the predefined mask matrixx and skip the blocks where mask is 1.


## Experiments and results

### Training time

### Model Accuracy



### Memory usage




## Limitation

## ignore point

## Links

[Video]()

[Notebook]()

[Paper]()

[Github repo with code]()

[Presentation reference]()

Application

* [AITemplate]() 
* [Nvidia's FasterTransformer]()
* Pytorch
* Huggingface






