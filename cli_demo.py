from transformers.generation.utils import logger

import argparse
import warnings
import torch
import os
import platform

from transformers import AutoTokenizer, AutoModelForCausalLM

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="OpenMEDLab/PULSE-20bv5", type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--input_max_len", default=1536, type=int)
parser.add_argument("--model_type", default='在线问诊', type=str)
parser.add_argument("--gen_max_length", default=512, type=int)
parser.add_argument("--top_k", default=6, type=int)
parser.add_argument("--top_p", default=0.1, type=float)
parser.add_argument("--temperature", default=0.7, type=float)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

# gen_max_length = args.gen_max_length
# top_k = args.top_k
# top_p = args.top_p
# temperature = args.temperature


first_instruction = "Instructions: You are Helper, a large language model full of intelligence. Respond conversationally."


model_type_prompt_map = {
    '医学知识QA': "若我是位患者，你是位资深医生，能够协助解答我的问题和疑虑，请为我提供回复。\n",
    '在线问诊': "假设你是一位经验丰富并且非常谨慎的的医生，会通过和患者进行多次的问答来明确自己的猜测，并且每次只能提一个问题，最终只会推荐相应的检验、检查、就诊科室以及疑似的诊断，请回复患者的提问：\n",
    'Base': "", 
}


assert args.model_type in model_type_prompt_map


def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def main():

    input_ids = tokenizer(
        first_instruction,
        add_special_tokens=False
    ).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]
    i = 0

    print("欢迎使用<|modelname|>！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")

    while True:
        query = input("User: ")
        query = query.strip() 

        if len(query) == 0:
            continue

        if query == "stop":
            break

        if query == "clear":
            clear()
            input_ids = tokenizer(
                first_instruction,
            ).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]
            i = 0
            continue

        if i == 0:
            query = model_type_prompt_map[args.model_type] + query

        input_ids += tokenizer("User: " + query).input_ids
        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]  

        # print(tokenizer.decode(input_ids, skip_special_tokens=False)) 

        model_inputs = tokenizer.pad(
            {"input_ids": [input_ids + tokenizer("Helper: ").input_ids[:1]]}, 
            return_tensors="pt",
        )

        inputs = model_inputs.input_ids[:,-args.input_max_len:]
        attention_mask = model_inputs.attention_mask[:,-args.input_max_len:]

        max_length = inputs.shape[1] + args.gen_max_length
        min_length = inputs.shape[1] + 1 # add eos

        outputs = model.generate(
            inputs=inputs.cuda(),
            attention_mask=attention_mask.cuda(),
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=1,
            eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        )

        outputs_token = outputs[0].tolist()

        new_start_pos = inputs.shape[1]
        new_end_pos = new_start_pos

        while new_end_pos < len(outputs_token) and outputs_token[new_end_pos] != tokenizer.convert_tokens_to_ids("</s>"):
            new_end_pos += 1

        outputs_token = list(tokenizer("Helper: ").input_ids[:1]) + outputs_token[new_start_pos:new_end_pos]

        input_ids += outputs_token
        input_ids += [tokenizer.convert_tokens_to_ids("</s>")] 

        otext = tokenizer.decode(
            outputs_token, 
            skip_special_tokens=False
        )

        print(otext)
        i += 1

            
    
if __name__ == "__main__":
    main()
















