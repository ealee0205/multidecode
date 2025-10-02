
import copy
import time
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
import torch
import transformers
import torch.nn.functional as F
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt

def mdecode(model=None,steer=None, input_ids=None, n_branch=1,tokens_to_add=10):
    """
    An optimized multi decoding algorithm for text generation.

    Args:
        model: The pre-trained language model.
        steer: A function that guides the text generation by selecting tokens.
        input_ids: The input token IDs.
        n_branch: The number of sequences to generate.
        tokens_to_add: The number of tokens to add to each branch.

    Returns:
        A list of lists, where each sublist contains the generated token IDs for a branch.
    """
    global_t0=time.time()
    assert model is not None,'Model is required argument'
    if steer is None:
        # default steer is just a greedy
        def __steer(branchs,logits,output):
            next_token_probs = F.softmax(logits / 0.7, dim=-1)
            tokens = torch.argmax(next_token_probs,dim=-1)
            return tokens
        steer = __steer

    batchsize,context_len=input_ids.shape
    # Each branch is a list of token positions
    branchs = [[[] for _ in range(n_branch)] for _ in range(batchsize)]
    pkv=None
    position_ids=torch.full((batchsize,n_branch),fill_value=context_len+1,dtype=torch.int).to(model.device)
    
    part1=torch.full((batchsize,1,n_branch,context_len+n_branch*tokens_to_add),fill_value=float('-inf')).to(model.device)
    part2=torch.where(torch.eye(n_branch) == 1, torch.tensor(0.0), torch.tensor(float('-inf')))\
            .unsqueeze(0).unsqueeze(0).expand(batchsize, 1, -1, -1).repeat(1, 1, 1, tokens_to_add).to(model.device)
    full_mask=torch.cat([part1,part2],dim=-1)

    for n in range(tokens_to_add):
        _,seq_len=input_ids.shape
        with torch.no_grad():
            if n==0: # For the first token, run the model on the context
                output=model.forward(input_ids=input_ids, past_key_values=pkv, use_cache=True)
                logits=output.logits[:,-1:,:].expand(-1,n_branch,-1)
                pkv = output.past_key_values
            else: # For subsequent tokens, use the cache and attention mask

                # position_ids=torch.full((batchsize,n_branch),fill_value=contnxtt_len+n).to(model.device)

                mask=torch.full((batchsize,1,n_branch,seq_len),fill_value=float('-inf')).to(model.device)
                mask[:,:,:,:context_len]=float(0)  # Allow attention to context
                for idx in range(batchsize):
                    for branch in range(n_branch):
                        for pos in range(n):
                            mask[idx,:,branch,context_len+branchs[idx][branch][pos]]=float(0.0)
                #mask = full_mask[:,:,:,:seq_len]

                output=model.forward(input_ids=input_ids[:,-n_branch:],position_ids=position_ids,attention_mask=mask, past_key_values=pkv, use_cache=True)
                logits=output.logits  # Get logits for the next token
                pkv = output.past_key_values   # Update past key/values
                position_ids+=1

            # external function to select tokens and steer the branches
            tokens=steer(branchs,logits,output)

            # #greedy
            # next_token_probs = F.softmax(output.logits / 0.7, dim=-1)
            # tokens = torch.argmax(next_token_probs,dim=-1)


            # add the selected tokens to the sequence
            input_ids = torch.cat([input_ids,tokens], dim=-1)

            # record location of branch
            for idx in range(batchsize):
                for branch in range(n_branch):
                    branchs[idx][branch].append(n*n_branch+branch)

    # Extract the generated branch ID
    branch_ids=[]
    for idx in range(batchsize):
        branch_ids.append([])
        for branch in range(n_branch):
            ids=input_ids[idx,torch.tensor(branchs[idx][branch])+context_len].to('cpu')
            branch_ids[idx].append(ids)
            
    return branch_ids