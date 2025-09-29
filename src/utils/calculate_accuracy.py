import json

possible_ids_medqa = json.load(open("data/processed/benchmark_open_filtered.json"))['medqa'].keys()
possible_ids_mmlu = json.load(open("data/processed/benchmark_open_filtered.json"))['mmlu'].keys()
possible_ids_medmcqa = json.load(open("data/processed/benchmark_open_filtered.json"))['medmcqa'].keys() 

def filter_completions(completions, subset):
    if subset == "medqa":
        possible_ids = possible_ids_medqa
    elif subset == "mmlu":  
        possible_ids = possible_ids_mmlu
    elif subset == "medmcqa":
        possible_ids = possible_ids_medmcqa
    else:
        raise ValueError(f"Unknown subset: {subset}")   
    filtered = [el for el in completions if el['id_question'] in possible_ids]
    return filtered

def calculate_accuracy(completions, mode):

    if mode == "roman_numeral":
        roman2letter = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D'}
        if completions[0]['final_answer'] in roman2letter and not completions[0]['gold_answer'] in roman2letter:
            correct = [roman2letter[el['gold_answer']] == el['final_answer'] for el in completions]
        else:
            correct = [el['gold_answer'] == el['final_answer'] for el in completions]

    elif mode == "yes_no_maybe":
        correct = [el['final_answer'].lower() == "yes" if el['final_answer'] else False  for el in completions]
    elif mode == "incorrect":
        correct = [el['gold_answer'] == el['final_answer'] for el in completions]
    else:
        correct = [el['gold_answer'].replace(".","") == el['final_answer'].replace(".","") if el['final_answer'] and el['gold_answer'] else False for el in completions]

    total = len(correct)
    accuracy = sum(correct) / total * 100 if total > 0 else 0
    return accuracy

datasets = ["medqa", "mmlu", "medmcqa"]
modes = ["incorrect", "roman_numeral", "fixed_pos", "no_symbols", "none_of_the_provided", "options_only", "yes_no_maybe"]
model_name = "together_api/openai/gpt-oss-120b" #"together_api/meta-llama/Llama-3.3-70B-Instruct-Turbo" #

output = []
for dataset in datasets:
    results = {}
    results["model_name"] = model_name
    results["dataset"] = dataset
    results["open"] = 0.0
    results["standard"] = 0.0
    for mode in modes:
        print(f"Calculating accuracy for mode: {mode} | dataset: {dataset}")
        with open(f'out/completions/{model_name}/{dataset}/generations_{mode}.jsonl', 'r') as f:
            completions = [json.loads(line) for line in f.readlines()]
        completions = filter_completions(completions, dataset)
        
        accuracy = calculate_accuracy(completions, mode)
        
        results[mode] = round(accuracy, 1)
        
        
        print(f"Accuracy for mode {mode}: {accuracy:.1f}%")
    output.append(results)

    with open(f'out/completions/{model_name}/accuracy_summary.jsonl', 'a') as f:
        for res in output:
            if res["dataset"] == dataset:
                f.write(json.dumps(res) + "\n")
    print("\n") 

    # convert to latex table

    import pandas as pd
    # load from jsonl
    with open(f'out/completions/{model_name}/accuracy_summary.jsonl', 'r') as f:
        output = [json.loads(line) for line in f.readlines()]
    
    df = pd.DataFrame(output)
    df.to_latex(f"out/completions/{model_name}/accuracy_summary.tex", index=False, float_format="%.1f")
    print("\n\n")