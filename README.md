# Flashattention_paper_presentation


## Background

### Problem Overview
Transformers have grown larger and deeper, but equipping them with longer context remains difficult, since the self-attention module at their heart
has time and memory complexity quadratic in sequence length. An important question is whether making attention faster and more memory-efficient can help Transformer models address their runtime and memory challenges for long sequences.

Many approximate attention methods have aimed to reduce the compute and memory requirements of attention. Many of them do not display time speedup against standard attention. One main reason is that they focus on FLOP reduction and tend to ignore overheads from memory access (IO).

### Methods
IO awareness with tiling and recomputing


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

