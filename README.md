# Flashattention_paper_presentation


## Background

### Problem Overview

Transformers have become larger and more complex, but incorporating longer context into them is still a challenging task because their self-attention module has a time and memory complexity that scales quadratically with sequence length. The problem this paper aimed to addresss is how to enhance the speed and memory efficiency of attention in order to assist Transformer models in overcoming their memory and runtime obstacles for long sequences.

Many other methods have been developed. However, many of these methods do not show a significant improvement in speed over standard attention because they primarily concentrate on reducing the number of floating-point operations (FLOPs) and neglect the overheads from memory access (IO).


### Hardware performance 

* SRAM vs HBM

![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/17782d32252de3ba7d854dbdfa9eb3108829bbb9/image1.png)



* Time calculation of operations
    * Computation-bound

    * Memory-bound: The time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g.,activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).


### Convert the problem
The goal is to maximize the use of fast memory: SRAM  In the meanwhile, the most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations. 


### Methods
IO aware with carefully accounting for the number of access to the slow and fast memory through tiling and recomputing




## FlashAttention

### Recall Standard Attention
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image2.png)



### Tech

* Tiling: 

  1. Split the attention: K, Q, V into blocks. 
  2. Load the blocks from HBM and transfer into SRAM. 


* Recomputation

  1. Store the scaling parameter
  2. Recompute the softmax result
  
  The goal of this part is to not store intermediate values in HBM for the backward pass, which reduce the number of memory access to HBM


### Algorithm


![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image3.png)



## Complexity Analysis on HBM access
Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑. Standard attention (Algorithm 0) requires Θ(𝑁 𝑑 + 𝑁2) HBM accesses, while FlashAttention (Algorithm 1) requires Θ(𝑁2𝑑2𝑀−1) HBM accesses.


* Standard Attetion

  * Load Q and K blocks: Θ(𝑁 𝑑) 
  * Write S = QK into HDM:  Θ(𝑁 2)
  * P = softmax(S), the input S is read from HBM and the output P is written to HBM: Θ(𝑁 2)
  * O = PV, readP, V from HBM and write O into HBM: Θ(𝑁 𝑑 + N2)
  In total: Θ(𝑁 𝑑 + N2)

* FlashatAttention
  
  Assume M is the size of SRAM. M should be larger than d smaller tha Nd

  * Load V and K blocks: Θ(𝑁 𝑑) in total
  * Every block, load corresponding Q block and O
    Notice: we can only load the total A blocks for one time, but every V/K block, we need to load the O allvover again. The access should be Θ (𝑁 𝑑 + 𝑁       𝑑𝑇𝑐) = Θ(𝑁 𝑑𝑇𝑐).
  * 𝐵𝑐 𝑑 = 𝑂(𝑀) ⇔ 𝐵𝑐 = 𝑂(M/d)
  * 𝑇𝑐 =𝑁/𝐵𝑐
  * Θ(𝑁 𝑑𝑇𝑐) = Θ(𝑁2𝑑2/𝑀）


## Extension: Block-Sparse FlashatAttention

Get the predefined mask matrix and skip the blocks where mask is 1.


## Experiments and results

### Training time

### Model Accuracy



### Memory usage




## Limitation

## ignore point
In the complexity analysis, even though author carefully calculate the memory access in HBM whe we read the K,Q and V. But the author ignored the memory access when we need to write the scaling l and m into the HBM. Compared with the memory level of the previous step, this part does not account for much, because they are all single-row N vectors, but memory bound still exists. 


## Links

[Video]()

[Notebook]()

[Paper]()

[Github repo with code]()

[Presentation reference]()

Application

* [AITemplate]() 
* [Nvidia's FasterTransformer]()
* [Pytorch]()
* [Huggingface]()






