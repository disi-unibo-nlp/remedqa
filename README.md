# ReMedQA

This repository contains the dataset and code for our paper: **"ReMedQA: Are We Done With Medical Multiple-Choice Benchmarks?"**

Medical MCQA benchmarks report near-human accuracy, but accuracy alone is not a reliable measure of competence. Models often change their answers under minor perturbations, revealing a lack of robustness. **ReMedQA** addresses this gap by augmenting standard medical MCQA datasets with **open-answer variants** and **systematically perturbed items**, enabling fine-grained evaluation of model reliability.

We also introduce two new metrics:  
- 🧠 **ReAcc** – measures correctness across all variations  
- 🔁 **ReCon** – measures consistency regardless of correctness

Our findings show that high accuracy can mask low reliability. Large models often exploit structural cues, while smaller models are underestimated by standard MCQA. Despite near-saturated accuracy, we are **not** yet done with medical multiple-choice benchmarks.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/remedqa.git
cd remedqa
pip install -r requirements.txt
```

## Quick Start

### Evaluating Open-Source Models (via vLLM)
```bash
MODEL_TYPE="mediphi"
SEED=42
BATCH_SIZE=8
MAX_NEW_TOKENS=2000
MODE="fixed_pos no_symbols none_of_the_provided" # select your modes

# ---- Loop over subsets ----
for SUBSET in medqa medmcqa mmlu; do

  if [ "$SUBSET" = "medqa" ]; then
    DATASET_PATH="data/bench/medqa_all.json"
  elif [ "$SUBSET" = "mmlu" ]; then
    DATASET_PATH="data/bench/mmlu_all.json"
  elif [ "$SUBSET" = "medmcqa" ]; then
    DATASET_PATH="data/bench/medmcqa_all.json"
  fi
  
  echo "Running for subset: $SUBSET"
  
  CMD="python3 -m src.bench_vllm \
    --model-type $MODEL_TYPE \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --max-new-tokens $MAX_NEW_TOKENS \
    --dataset-path $DATASET_PATH \
    --subset $SUBSET \
    --modes $MODE"

  echo "Executing: $CMD"
  eval $CMD
  echo "-------------------------------------"
done
```

### Evaluating API Models (OpenAI, Gemini, TogetherAI)

```bash
# example gemini 
SUBSET="mmlu"
MODEL_NAME="gemini-2.5-flash" #"gemini-2.5-flash" #"gpt-5-mini" #"openai/gpt-oss-120b" "meta-llama/Llama-3.3-70B-Instruct-Turbo" 
OUTPUT_DIR="out/completions"
OPTIONS_ONLY_MODE=false

echo "Evaluating API Models..."
python3 -m src.bench_api_models \
    --output-dir "$OUTPUT_DIR" \
    --subset "$SUBSET" \
    --model-name "$MODEL_NAME" \
    $(if $OPTIONS_ONLY_MODE; then echo "--options-only-mode"; fi)  
```

### Retrieving API Results

```bash
JOB_NAME="<your_batch_id>" 
OUTPUT_DIR="<your_output_dir>"
OPTIONS_ONLY_MODE=true
DATASET_DIR="data/bench"

echo "Retrieving API results and saving in: $OUTPUT_DIR"
python3 -m src.get_api_results \
    --job-name "$JOB_NAME" \
    -o "$OUTPUT_DIR" \
    --dataset-dir "$DATASET_DIR" \
    $(if $OPTIONS_ONLY_MODE; then echo "--options-only-mode"; fi)  
```

## Mapping Open-Ended Answers
To map open-ended answers back to the original multiple-choice options:

```bash
# example on MedQA
python3 src/llm_judge.py --subset medqa --completion_dir out/completions/mediphi
```

## Creating Input Prompts
Generate input prompts for each data mode:

```bash
OPTIONS_ONLY_MODE=false
INPUT="data/processed/benchmark_closed_filtered.json"
THINK_MODE=false

if $OPTIONS_ONLY_MODE; then
    echo "Creating options-only mode datasets..."


    for subset in "medqa" "medmcqa" "mmlu"; do
        OUTPUT="data/def/${subset}_options_only.json"
        TEMPLATE="data/templates/options_only_templates_think.json"
        echo "Creating data modes for subset: $subset"
        python3 -m src.utils.create_data_modes \
            -i "$INPUT" \
            -o "$OUTPUT" \
            --subset "$subset" \
            --options-only-mode \
            --template "$TEMPLATE"
    done

else
    OPEN_FILE="data/processed/benchmark_open_filtered.json"

    for subset in "medqa" "medmcqa" "mmlu"; do
        if $THINK_MODE; then
            OUTPUT="data/def/${subset}_all_think.json"
            TEMPLATE="data/templates/templates_think.json"
        else
            OUTPUT="data/def/${subset}_all.json"
            TEMPLATE="data/templates/templates.json"
        fi
        
        echo "Creating data modes for subset: $subset"
        python3 -m src.utils.create_data_modes \
            -i "$INPUT" \
            -o "$OUTPUT" \
            --subset "$subset" \
            --open-file "$OPEN_FILE" \
            --template "$TEMPLATE"
    done
fi
```

## Computing Evaluation Scores
Calculate ReAcc and ReCon scores:

```bash
MODEL_NAME="together_api/openai/gpt-oss-120b" #"openai_api/gpt-5-mini" #"together_api/openai/gpt-oss-120b" "llama3" "gemini_api/gemini-2.5-flash" #"together_api/meta-llama/Llama-3.3-70B-Instruct-Turbo"  "gemma3" "medgemma" "llama3" "Llama3-Med42-8B" "mediphi" "Phi-3.5-mini"
OPTIONS_ONLY_MODE=false

echo "Evaluating API Models..."
python3 -m src.bench_api_models \
    --output-dir "$OUTPUT_DIR" \
    --subset "$SUBSET" \
    --model-name "$MODEL_NAME" \
    $(if $OPTIONS_ONLY_MODE; then echo "--options-only-mode"; fi)  
```


## Data Preprocessing Pipeline

To reproduce the open-ended dataset creation process:

```python

# 1. use GPT-4.1 to convert each subset (MedQA, MedMCQA, MMLU) into open-ended version
process_data/convert_mcqa_open_api.py
   
# 2. parse OpenAI API completion
process_data/parse_mcqa_open_api.py

# 3. save separately for each subset parsed output
process_data/save_bench_open.py

# 4. combine all open-ended subset 
process_data/combine_bench_open.py

# 5. filter medmcqa to 1K samples across the previous open-ended converted question
process_data/filter_medmcqa.py

# 6. create the final versions of open-ended and closed mcqa dataset.
process_data/process_dataset.py
```

*Note: Check the input parameters within each script to adapt to your specific needs.*

---

## Results

### Overall Perfomance
<p align="center">
  <img src=".github/images/remedqa_abstract.png" alt="ReMedQA MCQA Consistency" style="max-width: 300px; width: 100%; height: auto;">
</p>

### Dataset-Specific Reliability
<p align="center">
  <img src=".github/images/remedqa_results.png" alt="ReMedQA Results">
</p>

### Consistency under Perturbations 
<p align="center">
  <img src=".github/images/remeqa_consistency_mcqa.png" alt="ReMedQA MCQA Consistency">
</p>