# Efficient MultiDecoding with Hugging Face Transformers

This repository explores how to use an optimized decoding algorithm for text generation with Hugging Face Transformers. We call this algorithm 'MultiDecode'. It leverages the `position_ids` and `attention_mask` arguments to the transformer's forward method to simultaneously and efficiently generate the next token for multiple independent sequences from a single shared KV cache.

## Introduction

Beam search is a popular decoding algorithm for text generation that explores multiple candidate sequences (branchs) to find the most likely output. However, traditional beam search can be computationally expensive, especially when generating long sequences or using a large number of beams.

MultiDecode is a generalization of beam search that can be applied to many use case, such as 
- beamsearch 
- multiple questions 
- writing in the margins
- parallel reasoning
- parallel sampling (i.e. entropix)
- predicting users 

MultiDecode is an optimized decoding algorithm that improves efficiency by processing multiple generative sequences simultaneously.  It does this by using the position_ids and attention_mask arguments to simultaneously predict the next token for all branches.  This is more efficient because the context sequence (prior to the branching) is only loaded once and is share amoungst all branchs. It is also faster because multiple tokens are generated on each forward pass of the model.
