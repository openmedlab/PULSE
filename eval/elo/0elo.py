# %%
import glob

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
    model_name = path.split("/")[-1].replace(".jsonl", "")

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

    all_predict_data[model_name] = elo_data


# %%
all_predict_data.keys()

# %%
elo_df = {}

# %%
from collections import defaultdict
import random

K=8
INIT_RATING=1000
SCALE=400
BASE=10


for task_name in all_predict_data.keys():

    rating = defaultdict(lambda: INIT_RATING)

    r_idxs = list(range(len(all_predict_data[task_name])))
    random.seed(42)
    random.shuffle(r_idxs)

    for r_idx in r_idxs:
        round_data_item = all_predict_data[task_name][r_idx]

        score = round_data_item['score']

        if score is None:
            continue

        # if round_data_item['model_a'] in {"PULSE_14bv5"}:
        #     continue

        # if round_data_item['model_b'] in {"PULSE_14bv5"}:
        #     continue

        ra = rating[round_data_item['model_a']]
        rb = rating[round_data_item['model_b']]

        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))

        if score[0] > score[1]:
            sa = 1
        elif score[1] > score[0]:
            sa = 0
        elif score[0] == score[1]:
            sa = 0.5

        rating[round_data_item['model_a']] += K * (sa - ea)
        rating[round_data_item['model_b']] += K * (1 - sa - eb)

    rating = sorted(rating.items(), key=lambda x:-x[1])

    for k,v in rating:
        if k not in elo_df:
            elo_df[k] = {item: None for item in all_predict_data.keys()}

        elo_df[k][task_name] = v

    print(task_name)
    print(rating)
    print("----------------------------------------")


# %%
rating = defaultdict(lambda: INIT_RATING)

all_data = []

for v in all_predict_data.values():
    all_data.extend(v)


r_idxs = list(range(len(all_data)))
random.seed(42)
random.shuffle(r_idxs)

for r_idx in r_idxs:
    round_data_item = all_data[r_idx]

    score = round_data_item['score']

    if score is None:
        continue

    # if round_data_item['model_a'] in {"PULSE_14bv5"}:
    #     continue

    # if round_data_item['model_b'] in {"PULSE_14bv5"}:
    #     continue

    ra = rating[round_data_item['model_a']]
    rb = rating[round_data_item['model_b']]

    ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
    eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))

    if score[0] > score[1]:
        sa = 1
    elif score[1] > score[0]:
        sa = 0
    elif score[0] == score[1]:
        sa = 0.5

    rating[round_data_item['model_a']] += K * (sa - ea)
    rating[round_data_item['model_b']] += K * (1 - sa - eb)

rating = sorted(rating.items(), key=lambda x:-x[1])

for k,v in rating:
    if k not in elo_df:
        elo_df[k] = {item: None for item in all_predict_data.keys()}

    elo_df[k]["ALL"] = v

# %%
import pandas as pd

model_size_map = {
    "GPT4": "220B*8(?)",
    "ChatGPT": "175B(?)",

    "PULSE_14bv5": "14B",
    "PULSE_7bv5": "7B",
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



