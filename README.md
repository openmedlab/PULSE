# PULSE

<p align="center" width="100%">
<img src="pics/PULSE.png" alt="MOSS" style="width: 60%; min-width: 300px; display: block; margin: 30px;">
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](./LICENSE)
[![Model License](https://img.shields.io/badge/Model%20License-GNU%20AGPL%203.0-red.svg)](./MODEL_LICENSE)

\[[中文版](./README.md)\] \[[English](./README_en.md)\] 

## 目录

- [模型](#模型)
  - [主要功能](#主要功能)
  - [下载地址](#下载地址)
  - [局限性](#局限性)
  - [Elo评测](#Elo评测)
  - [相关应用](#相关应用)
  - [用例](#用例)
- [推理](#推理)
  - [硬件要求](#硬件要求)
  - [下载安装](#下载安装)
  - [使用示例](#使用示例)
- [友情链接](#友情链接)
- [致谢](#致谢)
- [开源协议](#开源协议)

----

## 模型

### 主要功能
- **中文医疗大语言模型**
- **大规模训练**：PULSE模型使用约4,000,000个中文医学领域和通用领域的指令微调数据进行进一步调优。
- **全面的中文医学自然语言处理任务**：PULSE支持医学领域的各种自然语言处理任务，包括健康教育、医师考试问题、报告解读、医疗记录结构化以及模拟诊断和治疗。

### 下载地址

- [**PULSE-7b**](https://huggingface.co/OpenMEDLab/PULSE-7bv5)
- 我们将持续更新7B大小的模型，如果您需要更大的模型或者量化后的版本，欢迎联系 xujie@pjlab.org.cn 建立合作

### 局限性

- 由于模型参数量较小和自回归生成范式，尽管模型提供了有关疾病诊断和治疗的推理结果，但这些结果不能代替线下职业医生的建议和治疗方案。所有回答仅供参考，不应作为诊断或治疗的依据。我们强烈建议用户在需要诊断或治疗疾病时，寻求专业医生的帮助和建议。

### Elo评测
| model_name   | model_size   |     ALL |   MedQA_Mainland |   PromptCBLUE |   detailedMedQA |   webMedQA |
|:-------------|:-------------|--------:|-----------------:|--------------:|----------------------:|-----------:|
| GPT4         | 220B*8(?)    | 1243.79 |          1118.20 |       1166.39 |               1122.20 |    1132.74 |
| ChatGPT      | 175B(?)      | 1149.38 |          1083.05 |       1102.31 |               1098.38 |    1097.88 |
| PULSE_14b  | 14B          | 1114.23 |          1003.55 |       1055.56 |               1074.79 |    1074.28 |
| PULSE_7b   | 7B           | 1084.18 |          1047.35 |       1047.27 |               1029.37 |    1069.40 |
| QiZhenGPT    | 13B          |  979.94 |           952.66 |        929.56 |               1076.41 |    1006.85 |
| BianQue      | 6B           |  959.35 |           927.86 |        922.65 |               1050.22 |    1042.64 |
| Med-ChatGLM  | 6B           |  869.54 |           989.59 |        928.77 |                882.08 |     856.66 |
| BenTsao      | 7B           |  809.13 |           954.20 |        933.39 |                815.51 |     856.20 |
| DoctorGLM    | 6B           |  790.47 |           923.54 |        914.10 |                851.04 |     863.35 |


#### 评估方法
* 为了平衡成本，我们主要采用GPT4进行评估。如[QLoRA](https://arxiv.org/abs/2305.14314) 论证，单纯GPT4打分进行模型的对比随机波动性较大。这与我们的观察一致。因此采用了[QLoRA](https://arxiv.org/abs/2305.14314) 推荐的，现在比较普遍采用的Elo Rating tournament评测方法。

#### 评估数据集 [[eval/data]](eval/data)
* MedQA_Mainland: 从[MedQA](https://github.com/jind11/MedQA)的Mainland/test子集中抽150条
* PromptCBLUE: 从[PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE)的test子集中抽150条
* detailedMedQA: 由医学专业人士标注的98条常见医疗领域问题的详尽回答
* webMedQA: 从[webMedQA](https://github.com/hejunqing/webMedQA)的test子集中抽150条

#### 评测模型
* GPT4
* ChatGPT
* PULSE_14b
* [PULSE_7b](https://huggingface.co/OpenMEDLab/PULSE-7bv5)
* [QiZhenGPT](https://github.com/CMKRG/QiZhenGPT) (QiZhen-CaMA-13B-Checkpoint-6000)
* [BianQue](https://github.com/scutcyr/BianQue) (BianQue-2.0)
* [Med-ChatGLM](https://github.com/SCIR-HI/Med-ChatGLM)
* [BenTsao](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) (lora-alpaca-med-alpaca-alldata)
* [DoctorGLM](https://github.com/xionghonglin/DoctorGLM) (p-tuningv2)


#### 超参选择
* 出于成本考虑，我们选择每个数据集进行360轮随机评估，随机选择模型PK的先后顺序以抵消先后顺序的影响，随机种子为：42。Elo rating的实现代码和其他超参参照[Vicuna的Elo代码](https://raw.githubusercontent.com/lm-sys/FastChat/833d65032a715240a3978f4a8f08e7a496c83cb1/fastchat/serve/monitor/elo_analysis.py): K=8, init rating=1000。


### 相关应用

**XrayPULSE**

[openmedlab/XrayPULSE](https://github.com/openmedlab/XrayPULSE)

![image](./pics/XrayPULSE.png)

**病历结构化**

[JuneYaooo/llm_structure_tool](https://github.com/JuneYaooo/llm_structure_tool)

![image](./pics/llm_structure_tool.png)


**术语归一化**

[JOHNNY-fans/HierNorm](https://github.com/JOHNNY-fans/HierNorm)

![image](./pics/HierNorm.png) 

**知识库问答 (建设中)**

[JuneYaooo/medical_assistant](https://github.com/JuneYaooo/medical_kb_chatbot)

![image](./pics/medical_assistant.png)


### 简单用例

**健康科普**

![image](./pics/example_medical_science.png)

**医师考题**

![image](./pics/example_med_qa.png)

**报告解读**

![image](./pics/example_interpretation_report.png)

<!-- **病历结构化**

![image](./pics/example_structured_medical_record_0.png) -->

**模拟诊疗**

![image](./pics/example_automatic_consultation.png)

**医学无关问题无害处理**

![image](./pics/example_non_medical_issues.png)


## 推理
### 硬件要求

下表提供了一个batch size=1时本地部署PULSE进行推理所需的显存大小。

| 量化等级 | 加载模型 |
| -------- | -------- |
| FP16     | 14GB     |


### 下载安装
1. 下载本仓库内容至本地/远程服务器

```bash
git clone https://github.com/openmedlab/PULSE
cd PULSE
```

2. 创建conda环境安装依赖

```bash
conda env create -f llm.yml
conda activate llm
```

其中`torch`和`transformers`版本不建议低于推荐版本。

### 使用示例

#### 网页Demo

**Gradio**

```bash
python web_demo_gradio.py
```

#### 命令行Demo

您可以运行仓库中的`cli_demo.py`来启动一个简单的命令行Demo：

```bash
python cli_demo.py
```

## 友情链接

- [xxxxx]() - 关于PULSE及其相关技术的分享

如果您有其他开源项目使用或改进PULSE，欢迎提交Pull Request添加到README或在Issues中联系我们。

## 致谢

- 上海人工智能实验室
- 上海交通大学-清源研究院
- 华东理工大学-自然语言处理与大数据挖掘实验室


## 开源协议

本项目所含代码采用[Apache 2.0](https://github.com/openmedlab/PULSE/blob/main/LICENSE)协议，模型权重采用[GNU AGPL 3.0](https://github.com/openmedlab/PULSE/blob/main/MODEL_LICENSE)协议。如使用本项目所含模型及其修改版本提供服务产生误导性或有害性言论，造成不良影响，由服务提供方负责，与本项目无关。

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=openmedlab/PULSE&type=Date)](https://star-history.com/#openmedlab/PULSE&Date) -->