# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import os
import dotenv
import torch
dotenv.load_dotenv("../.env")
hf_token=os.getenv('HUGGINGFACE')

from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Load a pre-trained model and tokenizer for sequence-to-sequence tasks
model_name="meta-llama/Llama-3.2-1B" # Or any other suitable sequence-to-sequence model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Benchmark for transformers beam search.")

device='cuda:0'
model.to(device)

# Define a sample prompt
prompt1 = "Once upon a time,"
promptXL = "Once upon a time,"*200

# Tokenize the prompt
inputs1 = tokenizer(prompt1, return_tensors="pt").to(device)
inputsXL = tokenizer(promptXL, return_tensors="pt").to(device)

# Generate text using beam search
# num_beams controls the number of sequences to maintain
# do_sample=False disables sampling (default for beam search)
def benchmark(inputs,num_beams=150,num_avg=10):
    # warm up cache
    torch.cuda.empty_cache()
    output = model.generate(**inputs, num_beams=num_beams, do_sample=False,max_new_tokens=10,top_p=None,temperature=None,pad_token_id=128001)
    t0=time.time()
    for _ in range(num_avg):
        output = model.generate(**inputs, num_beams=num_beams, do_sample=False,max_new_tokens=10,top_p=None,temperature=None,pad_token_id=128001)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    t1=time.time()
    avg_t=(t1-t0)/num_avg
    print(f"Run time  context len={inputs['input_ids'].shape[1]}  {num_beams=} time {avg_t:.3f} sec")
nbeams=32

benchmark(inputs1,1)
benchmark(inputs1,nbeams)

benchmark(inputsXL,1)
benchmark(inputsXL,nbeams)
