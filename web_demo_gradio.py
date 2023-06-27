from transformers.generation.utils import logger
import mdtex2html
import gradio as gr
import argparse
import warnings
import torch
import os
from queue import Queue
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from typing import Optional

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="OpenMEDLab/PULSE-7bv5", type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--input_max_len", default=1536, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

first_instruction = "Instructions: You are Helper, a large language model full of intelligence. Respond conversationally."


model_type_prompt_map = {
    '医学知识QA': "若我是位患者，你是位资深医生，能够协助解答我的问题和疑虑，请为我提供回复。\n",
    '在线问诊': "假设你是一位经验丰富并且非常谨慎的的医生，会通过和患者进行多次的问答来明确自己的猜测，并且每次只能提一个问题，最终只会推荐相应的检验、检查、就诊科室以及疑似的诊断，请回复患者的提问：\n",
    'Base': "", 
}


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


class GradioStreamer(BaseStreamer):

    def __init__(
            self, 
            timeout: Optional[float] = None
        ):
        self.token_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value):
        list_value = value.tolist()
        if type(list_value[0]) == int:
            self.token_queue.put(list_value, timeout=self.timeout)

    def end(self):
        self.token_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


def predict(
    input, chatbot, 
    model_type, 
    gen_max_length, 
    top_p, top_k, temperature, 
    seed,
    history
):
    seed = int(seed)
    gen_max_length = int(gen_max_length)
    query = parse_text(input)

    assert len(query) > 0, "输入为长度为0"

    chatbot.append((query, None))
    history.append((query, None))

    input_ids = tokenizer(
        first_instruction,
        add_special_tokens=False
    ).input_ids + [tokenizer.convert_tokens_to_ids("</s>")]
    
    for i, (old_query, response) in enumerate(history):
        if i == 0:
            old_query = model_type_prompt_map[model_type] + old_query

        input_ids += tokenizer("User: " + old_query, add_special_tokens=False).input_ids
        input_ids += [tokenizer.convert_tokens_to_ids("</s>")]      

        if response is not None:
            input_ids += tokenizer("Helper: " + response, add_special_tokens=False).input_ids
            input_ids += [tokenizer.convert_tokens_to_ids("</s>")]

    # 引导启动
    input_ids += tokenizer("Helper: ", add_special_tokens=False).input_ids[:1]
    
    model_inputs = tokenizer.pad(
        {"input_ids": [input_ids]}, 
        return_tensors="pt",
    )

    inputs = model_inputs.input_ids[:,-args.input_max_len:]
    attention_mask = model_inputs.attention_mask[:,-args.input_max_len:]

    max_length = inputs.shape[1] + gen_max_length
    min_length = inputs.shape[1] + 1 # add eos

    streamer = GradioStreamer() # type: ignore

    if seed != -1:
        torch.manual_seed(seed)

    thread = Thread(target=model.generate, kwargs=dict(
        inputs=inputs.cuda(),
        attention_mask=attention_mask.cuda(),
        max_length=max_length,
        min_length=min_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        streamer=streamer,
    ))

    thread.start()

    # 起始
    output_tokens = list(tokenizer("Helper: ", add_special_tokens=False).input_ids[:1])
    for token in streamer:
        if token[0] not in {
            tokenizer.convert_tokens_to_ids("</s>"),
            tokenizer.convert_tokens_to_ids("<pad>"),
        }:
            output_tokens += token
            otext = tokenizer.decode(output_tokens, skip_special_tokens=False)

            if len(otext) > len("Helper: "):
                response = otext[len("Helper: "): ]

                chatbot[-1] = (query, parse_text(response))
                history[-1] = (query, response)

                yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">欢迎使用 <|modelname|> </h1>""")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot().style(height=530)

        with gr.Column(scale=1):
            model_type = gr.Radio(
                ['医学知识QA','在线问诊','Base'],
                label="Model Type",
                value='医学知识QA',
                interactive=True
            )
            gen_max_length = gr.Slider(
                1, 512, value=512, step=1.0, label="Generate Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.2, step=0.01,
                              label="Top P", interactive=True)
            
            top_k = gr.Slider(1, 50, value=9, step=1.0,
                              label="Top K", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)
            seed = gr.Number(label='Seed', value=-1)
            emptyBtn = gr.Button("Clear History")

    with gr.Column(scale=12):
        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=4).style(
            container=False)
    with gr.Column(min_width=32, scale=1):
        submitBtn = gr.Button("Submit", variant="primary")
        
    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, model_type, gen_max_length, top_p, top_k, temperature, seed, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)


demo.queue().launch(share=False, inbrowser=False)