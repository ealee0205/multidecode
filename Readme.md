# Generating tokens faster using predictions from multiple token positions

This repository shares how to unlock the existing parallel decoding ability of autoregressive large language models (LLMs).
We call this algorithm "MultiDecode".
Without any modification to the architecture, training, or hardware of the LLM, use cases involving multiple content blocks (such as RAG)
or multiple completion paths (such as beam search) can achieve almost linear speedup.
MultiDecode leverages custom RoPE position values and custom attention masks 
to simultaneously and efficiently generate exact next token predictions for multiple independent token positions, using a single, shared KV cache.
Support for these custom position and mask arguments already exists in many libraries, including the Hugging Face Transformers library.

This repo contains explanations, examples, and sample code showing how to use the MultiDecode paradigm for different use cases. 
A [YouTube video explanation of MultiDecode](https://youtu.be/9ld43ZYKzeI) is also available.

## Background

Autoregressive LLMs, such as Llama 3, decompose the text generation problem into a series of next token predictions,
with each prediction learning the conditional distribution of the token given all of the previous tokens in the sequence.
However, the self-attention mechanism commonly used in decoder-only transformer models does not exactly match this recurrent architecture.
Rather, self-attention performs pairwise comparisons of all elements in parallel across all token positions.
Self-attention is also position-agnostic, 
so position embeddings and triangular autoregressive masks are used to force it to model the linear sequence calculation.

The power of self-attention's ability to do parallel computation is commonly leveraged during training,
where teacher forcing is used for input tokens, and predictions from every token position are all used for loss calculation and learning.
During decoding, however, the common practice is to decode one token ata a time,
using only the prediction from the last token position.
The parallel nature of self-attention has been largely ignored for the inference task.
With MultiDecode, we look to open thinking to all of the parallel possibilities during decoding.

## MultiDecode

The key insight of this work is that if we think of tokens being nodes in a graph with edges between adjacent tokens,
then linear sequences are not the only graph that meet the autoregressive formulation requirements.
Below we show a linear sequence of tokens with whole number RoPE values 0 through 5. \
<img src="assets/images/sequence.png?raw=true" width="400">

If we introduce a branch in this graph, then each sequence from node 0 to one of the nodes numbered 5,
whether along the red branch or the blue branch, has the same properties as our simple linear sequence. \
<img src="assets/images/branch.png?raw=true" width="400">

In fact, given a tree, every path from the root to a leaf has the same properties as our simple linear sequence. \
<img src="assets/images/tree.png?raw=true" width="400">

It is also true that in a forest, every path from a root to a leaf is a sequence of tokens with consecutive whole numbers beginning with zero. \
<img src="assets/images/forest.png?raw=true" width="680">

The next token predictions for any leaf in a forest, conditioned on its ancestor nodes, will be the exact same calculation as if
only the tokens along the path from the root to the leaf had been given to the LLM as a linear sequence.

## Forming an input sequence

In order to input a forest of tokens into an LLM, the nodes (tokens) must be arranged into the standard one-dimensional input array.
An intuitive requirement is that tokens earlier in the causal chain for one or more other tokens 
should be placed physically earlier than the tokens with causal dependence on them.
Either a depth-first search or a breadth-first search (or a mix of them) of the forest is sufficient to meet this causal requirement.
We must assign custom RoPE embeddings to each node to match its height in its tree (instead of its physical position in the input),
and we must assign a custom mask so that each node can only attend to itself and its ancestors.
Given this configuration, we can read next token predictions from all of the leaves in parallel, 
and they will be the exact same calculation as if we had input each root-to-leaf sequence separately.
This is MultiDecoding.

## Beam search and other use cases

Beam search is a popular decoding algorithm for text generation that explores multiple candidate sequences (branchs) to find the most likely output. However, traditional beam search can be computationally expensive, especially when generating long sequences or using a large number of beams.

MultiDecode can make beam search dramatically faster by parallelizing multiple branches of the search at almost identical cost to standard decoding of only a single token.
MultiDecode speedup can be applied to many use cases, such as 
- beamsearch 
- multiple questions 
- writing in the margins
- parallel reasoning
- parallel sampling (i.e. entropix)
- predicting users 

MultiDecode is an optimized decoding algorithm that improves efficiency by processing multiple generative sequences simultaneously.  It does this by using the position_ids and attention_mask arguments to simultaneously predict the next token for all branches.  This is more efficient because the context sequence (prior to the branching) is only loaded once and is share amoungst all branchs. It is also faster because multiple tokens are generated on each forward pass of the model.
