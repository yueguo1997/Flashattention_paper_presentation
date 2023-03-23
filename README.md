# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness


## Background

### Problem Overview

Transformers have become larger and more complex, but incorporating longer context into them is still a challenging task because their self-attention module has a time and memory complexity that scales quadratically with sequence length. The problem this paper aimed to addresss is how to enhance the speed and memory efficiency of attention in order to assist Transformer models in overcoming their memory and runtime obstacles for long sequences.

Many other methods have been developed. However, many of these methods do not show a significant improvement in speed over standard attention because they primarily concentrate on reducing the number of floating-point operations (FLOPs) and neglect the overheads from memory access (IO).


### Hardware performance 

* SRAM vs HBM

![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/17782d32252de3ba7d854dbdfa9eb3108829bbb9/image1.png)

   For example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming         multiprocessors with bandwidth estimated around 19TB/s.


* Time calculation of operations
    * Computation-bound:the time taken by the operation is determined by how many arithmetic operations![image](https://user-images.githubusercontent.com/89158310/227127445-a7aa2a4a-d68d-4125-b7f9-f8bbe1b64949.png)

    * Memory-bound: 
      The time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. 
   
    Question:
    Is attention Computation-bound or Memory-bound? 


### Convert the problem
The goal is to maximize the use of fast memory: SRAM. In the meanwhile, the most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations. 


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


   ![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/417cbb5e26bd617cb6bfe50c4cbfbc7933d6b8af/Screen%20Shot%202023-03-21%20at%209.35.30%20PM.png)

  1. Store the scaling parameter instead of store the large intermediate matrix into HBM
  2. Recompute the softmax result

     
  
  The goal of this part is to not store intermediate values in HBM for the backward pass, which reduce the number of memory access to HBM


### Algorithm


![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/c49e4ad099f488b020e449a2b4ac9263ee9747f0/image3.png)



## Complexity Analysis on HBM access
Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑. Standard attention (Algorithm 0) requires Θ(𝑁 𝑑 + 𝑁2) HBM accesses, while FlashAttention (Algorithm 1) requires Θ(𝑁2𝑑2𝑀−1) HBM accesses.


* Standard Attention

  * Load Q and K blocks: $$\Theta(Nd)$$
  * Write S = QK into HDM:  $$\Theta(N^2)$$
  * P = softmax(S), the input S is read from HBM and the output P is written to HBM: $$\Theta(N^2)$$
  * O = PV, readP, V from HBM and write O into HBM: $$\Theta(Nd + N^2)$$
  In total: $$\Theta(Nd + N^2)$$

* FlashatAttention
  
  Assume M is the size of SRAM. M should be larger than d smaller than Nd

  * Load V and K blocks: $$\Theta(Nd)$$ in total
  * Every block, load corresponding Q block and O
    Notice: we can only load the total A blocks for one time, but every V/K block, we need to load the O allvover again. The access should be $$\Theta(𝑁 𝑑 + 𝑁𝑑T_c) = \Theta(NdT_c)$$.
  * $$d𝐵_c = 𝑂(𝑀) ⇔ 𝐵𝑐 = 𝑂(M/d)$$
  * $$T_c =𝑁/B_c$$
  * $$\Theta(𝑁𝑑T_c) = Θ(𝑁^2𝑑^2/𝑀)$$


## Extension: Block-Sparse FlashatAttention

Get the predefined mask matrix and skip the blocks where mask is 1.


## Experiments and results

### Training time
* Bert Model
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/da7c230dcb62b3581253fbad3642454baf0f24a3/Screen%20Shot%202023-03-21%20at%209.21.36%20PM.png)

* GPT Model
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/417cbb5e26bd617cb6bfe50c4cbfbc7933d6b8af/Screen%20Shot%202023-03-21%20at%209.21.46%20PM.png)

### Model Accuracy
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/417cbb5e26bd617cb6bfe50c4cbfbc7933d6b8af/Screen%20Shot%202023-03-21%20at%209.22.07%20PM.png)


### Memory usage
![plot](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/417cbb5e26bd617cb6bfe50c4cbfbc7933d6b8af/Screen%20Shot%202023-03-21%20at%209.22.56%20PM.png)


## Limitation
* Single GPU
* Low level language

## ignore point
In the complexity analysis, even though author carefully calculate the memory access in HBM whe we read the K,Q and V. But the author ignored the memory access when we need to write the scaling l and m into the HBM. Compared with the memory level of the previous step, this part does not account for much, because they are all single-row N vectors, but memory bound still exists. 


## Links

[Video](https://drive.google.com/file/d/1ifDY1vJ79rDCfUspbb3kblEcLv5faysq/view?usp=sharing)

[Notebook](https://github.com/yueguo1997/Flashattention_paper_presentation/blob/1fec30343ba23af8ba731d763efafac25f6693b3/Flashattention.ipynb)

[Paper](https://arxiv.org/pdf/2205.14135.pdf)

[Github repo with code](https://github.com/HazyResearch/flash-attention)


Application

* [AITemplate](https://ai.facebook.com/blog/gpu-inference-engine-nvidia-amd-open-source/) 
* [Nvidia's FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
* [Huggingface](https://github.com/huggingface/transformers)






