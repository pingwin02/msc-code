import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3-4B-Thinking-2507"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    raw_text = "Our first task is to familiarize ourselves with the framework we gonna use through most of the laboratories."
    input_ids = tokenizer(raw_text, return_tensors="pt").input_ids.to(device)

    print(f"Amount of tokens: {input_ids.shape[1]}")
    print(f"Decoded prompt: {tokenizer.decode(input_ids[0])}\n")

    generation_kwargs = {
        "min_new_tokens": 50,
        "max_new_tokens": 50,
        "do_sample": False,
    }

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(input_ids, **generation_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    duration = end_time - start_time

    generated_ids = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("===== Generated Text =====")
    print(generated_text)
    print("==========================")
    print(f"Generation time: {duration:.2f}s")
    print(f"Throughput: {len(generated_ids) / duration:.2f} tokens/s")

if __name__ == "__main__":
    main()
