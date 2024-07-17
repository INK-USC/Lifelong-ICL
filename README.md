# Lifelong-ICL: Stress-Testing Long-Context Language Models with Lifelong ICL and Task Haystack


This repository contains code for our paper Stress-Testing Long-Context Language Models with Lifelong ICL and Task Haystack. Lifelong ICL challenges long-context language models to learn various language tasks sequentially through in-context learning.

 We benchmark 10 open-source models and 2 commercial models using Lifelong ICL. Here are our main results:
| Model  / Pass Rate             | 1-shot 16-tasks (4k) | 2-shot 16-tasks (8k) | 4-shot 16-tasks (16k) | 8-shot 16-tasks (32k) | 2-shot 8-tasks (4k) | 2-shot 16-tasks (8k) | 2-shot 32-tasks (15k) | 2-shot 64-tasks (25k) |
| ------------------- | -------------------- | -------------------- | --------------------- | --------------------- | ------------------- | -------------------- | --------------------- | --------------------- |
| Mistral-7B (32k)    | 91.2                 | 73.8                 | 67.5                  | 47.5                  | 80.0                | 73.8                 | 72.5                  | 75.6                  |
| FILM-7B (32k)       | 77.5                 | 77.5                 | 72.5                  | 55.0                  | 87.5                | 77.5                 | 88.1                  | 75.3                  |
| Llama2-7B (32k)     | 77.5                 | 53.8                 | 41.2                  | -                     | 65.0                | 53.8                 | 59.4                  | 63.1                  |
| Llama2-7b (80k)     | 100.0                | 100.0                | 96.3                  | 76.3                  | 97.5                | 100.0                | 91.2                  | 89.7                  |
| Llama3-8B (1048k)   | 78.8                 | 76.2                 | 71.3                  | 57.5                  | 75.0                | 76.2                 | 75.6                  | 81.2                  |
| Llama3-70B (1048k) | 68.8                  | 50.0                 | 57.5                  | 51.2                  | 45.0                | 50.0                 | 59.4                  | 70.3 
| Yi-6B (200k)        | 61.3                 | 51.2                 | 43.8                  | 38.8                  | 50.0                | 51.2                 | 63.7                  | 65.6                  |
| Yi-9B (200k)        | 71.2                 | 71.2                 | 63.7                  | 47.5                  | 62.5                | 71.2                 | 61.3                  | 61.3                  |
| Yi-34B (200k)       | 62.5                 | 60.0                 | 63.8                  | 53.8                  | 87.5                | 60.0                 | 63.1                  | 59.4                  |
| Cmd-R-35B (128k)    | 81.2                 | 61.3                 | 58.8                  | 41.2                  | 82.5                | 61.3                 | 73.1                  | 77.2
| GPT-3.5-Turbo (16k) | 73.8                 | 71.3                 | 62.5                  | -                     | -                   | -                    | -                     | -                     |
| GPT-4o (128k)       | 86.3                 | 81.3                 | 83.8                  | 88.8                  | -                   | -                    | -                     | -                     |
- While long-context models excel at retrieving and pasting information within lengthy contexts, their ability to fully exploit the contextual information remains limited. 
- State-of-the-art closed models such as GPT-4o still struggle in this setting, failing 12.5% of the cases on average. And all open models we evaluate further lack behind by a large margin.

## Configure Environment

```bash
## Create a conda env
conda create -n llicl python=3.9
conda activate llicl
pip install pandas matplotlib scikit-learn retrying

## vLLM
pip install vllm==0.5.0

# (Optional) HF datasets
pip install datasets==2.18.0
pip install -U huggingface_hub

## (Optional) Openai API
pip install openai==1.25.1
```

## Data Preparation
Download and preprocess data:
```bash
cd preprocessing/tasks
bash run.sh
```
## Run Evaluation Pipeline
### Setup vllm_configs
This code uses vLLM as the inference framework. Configure your settings in `model/vllm_config.py`. eg:
```python
if model_name.lower()== "": # your model name
    model_config = {
        "model": "", # hf model identifier
        "max_model_len": "" # max length
    }
```
### Setup run.sh and start evaluation
Set model configuration in `scripts/evaluate/run_baseline.sh`:
```bash
MODEL_NAME="" # your model name
```
Start the baseline experiments:
```bash
bash run_baseline.sh
```
Run scaling shot and scaling tasks experiments:
```bash
bash run_scale_shot.sh
bash run_scale_task.sh
```

### Visualize results
Visualize results of scaling shot and scaling tasks experiments by using `playground/analysis_niath.ipynb`:
```python
# set your baseline and recall results directory
baseline_dir = "<your path>/output/default/baseline/<your model name>/ntask_nshot"
recall_dir = "<your path>/output/default/recall/<your model name>"
```
Run the respective Jupyter notebook to view the results. Example of visualization:
![example](https://github.com/cherry979988/Lifelong-ICL/assets/77602921/a37e9bd9-b65f-408d-a33f-5204b4f2e6b9)


Generate diagnostic reports by using `playground/analysis_diagnose.ipynb`:
```python
# set your results file path
baseline_results = "<your path>/output/default/baseline/<your model name>/ntask_nshot/results.csv" 
llicl_dir = "<your path>/output/default/recall/<your model name>"
n_shot = 2
n_task = 32
```

## (Optional) Run Controlled Experiments
Configure your model in `model/vllm_configs.py` and use the provided scripts for controlled experiments:
- `run_repeat.sh`: Repeat in-context learning demonstrations of one task for N times.
- `run_paraphrase.sh`: Use paraphrases instructions.
- `run_irrelevant.sh`: Prepend irrelevant text to n-shot in-context learning demonstrations.
- `run_control.sh`: Conduct replay and repeat experiments.
