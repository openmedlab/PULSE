# %%
import glob
import pandas as pd

# %%
elo_data_path = "eval/elo/elo_data/"
elo_outputs_path = "eval/elo/gpt4_elo_outputs/"

all_predict_paths = sorted(glob.glob(elo_data_path + "*.jsonl"))

# %%
def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        print(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return None

# %%
import json

all_predict_data = {}


for path in all_predict_paths:
    model_name = path.split("/")[-1].replace("_0629.jsonl", "").replace("_0628_2.jsonl", "").replace("_0628.jsonl", "").replace(".jsonl", "")

    elo_data = []
    elo_outputs = []
     
    with open(path, "r", encoding="utf8") as f_elo_data:
        for line in f_elo_data:
            line = line.rstrip()

            if len(line) == 0:
                continue

            line = json.loads(line)

            elo_data.append(line)

    with open(path.replace(elo_data_path, elo_outputs_path), "r", encoding="utf8") as f_elo_outputs:

        for line in f_elo_outputs:
            line = line.rstrip()

            if len(line) == 0:
                continue

            line = json.loads(line)['output']
            if line is None or len(line) == 0:
                elo_outputs.append(None)

            else:
                line = json.loads(line)
                elo_outputs.append(parse_score(line['content']))
    
    assert len(elo_data) == len(elo_outputs)

    for elo_data_item, elo_output in zip(elo_data, elo_outputs):
        elo_data_item['score'] = elo_output

        if elo_output is None:
            elo_data_item['winner'] = "tie (None)"
        elif elo_output[0] > elo_output[1]:
            elo_data_item['winner'] = "model_a"
        elif elo_output[1] > elo_output[0]:
            elo_data_item['winner'] = "model_b"
        elif elo_output[0] == elo_output[1]:
            elo_data_item['winner'] = "tie"

    if model_name not in all_predict_data:
        all_predict_data[model_name] = []


    for elo_data_item in elo_data:

        if elo_data_item['model_a'] in {"PULSE_7bv7", "PULSE_14bv7"}:
            continue

        if elo_data_item['model_b'] in {"PULSE_7bv7", "PULSE_14bv7"}:
            continue

        all_predict_data[model_name].append(elo_data_item)


# %%
del all_predict_data["med_detailed_answer"]
del all_predict_data["chunyu"]

# %%
elo_df = {}

# %%
from elo_analysis import report_elo_analysis_results

for task_name in all_predict_data.keys():

    tt_report = report_elo_analysis_results(all_predict_data[task_name])

    for k,v in tt_report['elo_rating_median'].items():
        if k not in elo_df:
            elo_df[k] = {item: None for item in all_predict_data.keys()}

        elo_df[k][task_name] = v

    print(task_name)
    print(tt_report['elo_rating_median'])
    print("----------------------------------------")


# %%
all_data = []

for v in all_predict_data.values():
    all_data.extend(v)

all_report = report_elo_analysis_results(all_data)


for k,v in all_report["elo_rating_median"].items():
    if k not in elo_df:
        elo_df[k] = {item: None for item in all_predict_data.keys()}

    elo_df[k]["ALL"] = v

# %%
import pandas as pd

model_size_map = {
    "GPT4": "220B*8(?)",
    "ChatGPT": "175B(?)",
    "ChatGLM": "6B",
    'MedicalGPT-zh': "6B",
    'Chinese-LLaMA-Alpaca-Plus-13b': "13B",

    "PULSE_14bv5": "14B",
    "PULSE_14bv5_add_prompt": "14B",
    "PULSE_7bv5": "7B",
    "PULSE_7bv5_add_prompt": "7B",

    "PULSE_14bv7": "14B",
    "PULSE_7bv7": "7B",

    "QiZhenGPT": "13B",
    "BianQue": "6B",
    "Med-ChatGLM": "6B",
    "BenTsao": "7B",
    "DoctorGLM": "6B",
}

elo_df_use = pd.DataFrame([{
    "model_name": k, 
    "model_size": model_size_map[k],
    "ALL": v["ALL"],
    **{dk:dv for dk,dv in v.items() if dk != "ALL"}
} for k,v in elo_df.items()])
# elo_df_use.columns = ['model_name', 'MedQA_Mainland', 'PromptCBLUE','chunyu.jsonl', 'med_detailed_answer.jsonl', 'webMedQA.jsonl', 'ALL']

# %%
print(elo_df_use.sort_values("ALL", ascending=False).to_markdown(index=False, floatfmt='.02f'))

# %%


# %%


# %%


# %%



