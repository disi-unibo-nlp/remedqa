import json
import itertools
import os

possible_ids_medqa = json.load(open("data/processed/benchmark_open_filtered.json"))['medqa'].keys()
possible_ids_medmcqa = json.load(open("data/processed/benchmark_open_filtered.json"))['medmcqa'].keys() 
possible_ids_mmlu = json.load(open("data/processed/benchmark_open_filtered.json"))['mmlu'].keys()
possible_ids_mmlu_dict = {}

for key in possible_ids_mmlu:
    subset = key.split("-", 1)[0]
    if subset not in possible_ids_mmlu_dict:
        possible_ids_mmlu_dict[subset] = []
    possible_ids_mmlu_dict[subset].append(key)

def generate_combinations_with_open(items, required="open"):
    """
    Generate all possible combinations of items that must include `required`,
    with group sizes ranging from 2 up to len(items)-1.
    
    Returns a list of tuples.
    """
    if required not in items:
        raise ValueError(f"Required element '{required}' not in list")

    others = [x for x in items if x != required]
    all_combos = []
    for r in range(2, len(others)+1):  # choose at least 1 other, at most len(others)
        for combo in itertools.combinations(others, r):
            all_combos.append((required,) + combo)
    return all_combos

def calculate_overall_accuracy(accuracy_dict, only_closed=False, closed_list=None):
    consistent = []
    from collections import Counter
    fails = []
    for key, answers in accuracy_dict.items():
        if only_closed:
            all_answers = [el['answer'] for el in answers if el['mode'] != "open" and el['mode'] in list(closed_list)]
        else:
            all_answers = [el['answer'] for el in answers if el['mode'] in list(closed_list)]
        #print(all_answers)
       
        #print(answer_dict)
        consistent.append(sum(all_answers) == len(all_answers))
    #print("Fails:", fails)
    total = len(consistent)
    overall_accuracy = sum(consistent) / total * 100 if total > 0 else 0
    return overall_accuracy

def calculate_overall_consistency(consistency_dict, only_closed=False, closed_list=None):
    consistent = []
    from collections import Counter
    fails = []
    for key, answers in consistency_dict.items():
        if only_closed:
            all_answers = [el['answer'] for el in answers if el['mode'] != "open" and el['mode'] in list(closed_list)]
        else:
            all_answers = [el['answer'] for el in answers if el['mode'] in list(closed_list)]
        #print(all_answers)
        try: 
            answer_dict = dict(Counter(all_answers)) if all_answers else {}
        except:
            fails.append(key)
            print("Fail:", all_answers)
            continue
        #print(answer_dict)
        consistent.append(len(answer_dict.keys()) == 1)
    print("Fails:", fails)
    total = len(consistent)
    overall_consistency = sum(consistent) / total * 100 if total > 0 else 0
    return overall_consistency

def filter_completions(completions, subset):
    if subset == "medqa":
        possible_ids = possible_ids_medqa
    elif subset == "medmcqa":
        possible_ids = possible_ids_medmcqa
    elif subset in ['professional_medicine', 'college_medicine', 'anatomy', 'clinical_knowledge', 'college_biology', 'medical_genetics']:  
        possible_ids = possible_ids_mmlu_dict[subset]
    else:
        raise ValueError(f"Unknown subset: {subset}")   
    filtered = [el for el in completions if el['id_question'] in possible_ids]
    return filtered

def calculate_consistency(closed_completions, open_completions_dict, mode, dataset_gold):
    consistent = []
    mapped_final_answers = []
    for completion in closed_completions:
        id_completion = completion['id_question']
        final_answer_closed = completion['final_answer']
        if id_completion in open_completions_dict and final_answer_closed:
            open_completion = open_completions_dict[id_completion]
            final_answer_open = open_completion['final_answer']
        else:
            continue
        
        if mode == "roman_numeral":
            roman2letter = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D'}
            if final_answer_closed in roman2letter:
                final_answer_closed = roman2letter[final_answer_closed]
            consistent.append(final_answer_open == final_answer_closed)
            mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_closed)})
            
        elif mode == "incorrect":
            #print(final_answer_closed, final_answer_open)
            if final_answer_closed and len(final_answer_closed) == 3:
                consistent.append(final_answer_open not in final_answer_closed)
                mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_open)})
            else:
                consistent.append(False)
                mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_closed)}) # randomly select the first, it won't be consistent
            
        elif mode == "fixed_pos":
            # map answer id to option value
            dataset_fixed_pos = dataset_gold[mode]
            id2option_fixed_pos = dataset_fixed_pos[id_completion]['options']
            #option2id_fixed_pos = {v: k for k, v in id2option_fixed_pos.items()}
            mapped_answer_closed = id2option_fixed_pos[final_answer_closed] if final_answer_closed in id2option_fixed_pos else ""
            mapped_answer_closed = mapped_answer_closed.replace(".","").replace("[", "").replace("]","")
            # orginal dataset
            dataset_open = dataset_gold['mcq']
            id2option_open = dataset_open[id_completion]['options']
            option2id_open = {v.replace(".","").replace("[", "").replace("]",""): k for k, v in id2option_open.items()}
            #print(final_answer_open)
            if not isinstance(final_answer_open, list) and final_answer_open in id2option_open:
                mapped_answer_open = id2option_open[final_answer_open]
                
                mapped_answer_open = mapped_answer_open.replace(".","").replace("[", "").replace("]","")

                consistent.append(mapped_answer_open == mapped_answer_closed)
                
            else:
                consistent.append(False)
            answer = str(option2id_open[mapped_answer_closed]) if mapped_answer_closed in option2id_open else ""
            # get the mapping towards the original set 
            mapped_final_answers.append({"id_question": id_completion, "answer": answer})

        elif mode == "no_symbols":
            final_answer_closed = final_answer_closed.replace(".","").replace("[", "").replace("]","")
            dataset_open = dataset_gold['mcq']
            id2option_open = dataset_open[id_completion]['options']
            option2id_open = {v.replace(".","").replace("[", "").replace("]",""): k for k, v in id2option_open.items()}
            if not isinstance(final_answer_open, list) and final_answer_open in id2option_open:
                mapped_answer_open = id2option_open[final_answer_open].replace(".","").replace("[", "").replace("]","")
                consistent.append(mapped_answer_open == final_answer_closed)
            else:
                consistent.append(False)
            # get the mapping towards the original set 

            final_answer_closed = option2id_open[final_answer_closed] if final_answer_closed in option2id_open else ""
            mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_closed)})
        else:
            consistent.append(final_answer_open == final_answer_closed)
            mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_closed)})

        
    total = len(consistent)
    consistency = sum(consistent) / total * 100 if total > 0 else 0
    
    return consistency, mapped_final_answers

def calculate_accuracy(completions, mode):
   
    if mode == "roman_numeral":
        roman2letter = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D'}
        if completions[0]['final_answer'] in roman2letter and not completions[0]['gold_answer'] in roman2letter:
            correct = [{'id_question': el['id_question'], 'answer': roman2letter[el['gold_answer']] == el['final_answer']} for el in completions]
        else:
            correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'] == el['final_answer']} for el in completions]

    elif mode == "yes_no_maybe":
        correct = [{'id_question': el['id_question'], 'answer': el['final_answer'].lower() == "yes" if el['final_answer'] else False} for el in completions]
    elif mode == "incorrect":
        correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'] == el['final_answer']} for el in completions]
    elif mode =="idk_answer":
        correct = []
        total_idk = 0
        for el in completions:

            if el['final_answer'].upper() == "E":
                total_idk += 1
                correct.append(0)

            elif el['gold_answer'] == el['final_answer']:
                correct.append(1)

            else: #el['gold_answer'] != el['final_answer']
                correct.append(-1)
        
        print("Number of idk answers:", total_idk)

    elif mode=="open":
        correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'].replace(".","").replace("[","").replace("]","") == el['final_answer'].replace(".","").replace("[","").replace("]","") or (el['valid_alternative'] == "Yes") if el['final_answer'] and el['gold_answer'] and not isinstance(el['final_answer'], list) else False} for el in completions]
    else:
        correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'].replace(".","") == el['final_answer'].replace(".","") if el['final_answer'] and el['gold_answer'] else False} for el in completions]

    correct_list = [el['answer'] for el in correct]
    total = len(correct_list)
    accuracy = sum(correct_list) / total * 100 if total > 0 else 0
    return accuracy, correct

# def calculate_accuracy(completions, mode):
   
#     if mode == "roman_numeral":
#         roman2letter = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D'}
#         if completions[0]['final_answer'] in roman2letter and not completions[0]['gold_answer'] in roman2letter:
#             correct = [roman2letter[el['gold_answer']] == el['final_answer'] for el in completions]
#         else:
#             correct = [el['gold_answer'] == el['final_answer'] for el in completions]

#     elif mode == "yes_no_maybe":
#         correct = [el['final_answer'].lower() == "yes" if el['final_answer'] else False  for el in completions]
#     elif mode == "incorrect":
#         correct = [el['gold_answer'] == el['final_answer'] for el in completions]
#     elif mode =="idk_answer":
#         correct = []
#         total_idk = 0
#         for el in completions:

#             if el['final_answer'].upper() == "E":
#                 total_idk += 1
#                 correct.append(0)

#             elif el['gold_answer'] == el['final_answer']:
#                 correct.append(1)

#             else: #el['gold_answer'] != el['final_answer']
#                 correct.append(-1)
        
#         print("Number of idk answers:", total_idk)

#     elif mode=="open":
#         correct = [el['gold_answer'].replace(".","").replace("[","").replace("]","") == el['final_answer'].replace(".","").replace("[","").replace("]","") or (el['valid_alternative'] == "Yes") if el['final_answer'] and el['gold_answer'] and not isinstance(el['final_answer'], list) else False for el in completions]
#     else:
#         correct = [el['gold_answer'].replace(".","") == el['final_answer'].replace(".","") if el['final_answer'] and el['gold_answer'] else False for el in completions]

    
#     total = len(correct)
#     accuracy = sum(correct) / total * 100 if total > 0 else 0
#     return accuracy

datasets = ["medqa", "medmcqa",'professional_medicine', 'college_medicine', 'anatomy', 'clinical_knowledge', 'college_biology', 'medical_genetics']#["medqa"]#, "medmcqa", 
modes = ["open", "mcq", "incorrect", "roman_numeral", "fixed_pos", "no_symbols","none_of_the_provided"]  #"idk_answer"]#]#, "options_only"]#, "yes_no_maybe"]#, ]
model_name = "llama3" #"openai_api/gpt-5-mini" #"gemini_api/gemini-2.5-flash"#"llama3" # #"together_api/openai/gpt-oss-120b" # #"medgemma" # # # #"openai_api/gpt-5-mini" #  # # #"openai_api/gpt-5-mini" #"gemini_api/gemini-2.5-flash" # #"together_api/openai/gpt-oss-120b" # #

output = []
id2open_answer = {}
consistency_dict = {}
accuracy_dict = {}
overall_results = {"model": model_name.split("/")[-1]}
for dataset in datasets:
    
    dataset_dir_name = dataset if dataset in ["medqa", "medmcqa"] else "mmlu"
    with open(f"data/processed/{dataset_dir_name}_all.json") as f:
        dataset_gold = json.load(f)#[args.subset]

    results = {}
    results["model_name"] = model_name
    results["dataset"] = dataset
    #results["open"] = 0.0
    #results["standard"] = 0.0
    for mode in modes:
        
        #dataset_gold_mode = dataset_gold[mode]
        print(f"Calculating accuracy for mode: {mode} | dataset: {dataset}")
        with open(f'out/completions/{model_name}/{dataset_dir_name}/generations_{mode}.jsonl', 'r') as f:
            completions = [json.loads(line) for line in f.readlines()]
        completions = filter_completions(completions, dataset)
        
        #if dataset == "mmlu":
            #subsets = set([key.split("-")[0] for key, el in dataset_gold[dataset].items()])
            #completions_list = []
            #for sub in subsets:
                #completions = [el for el in completions if el['id_question'].startswith(sub)]
                #for el in completions:
                #   el['dataset'] = sub
                #completions_list.append(completions)
        #else:
            #completions_list = [completions]

        if mode == "open":
            id2open_answer = {item['id_question']: item for item in completions}

        accuracy, final_answers = calculate_accuracy(completions, mode)
        
        results[mode] = round(accuracy, 1)
        
        print(f"Accuracy for mode {mode}: {accuracy:.1f}%")

        for answer in final_answers:
            id_answer = answer['id_question']
            answer_content = answer['answer']
            if id_answer not in accuracy_dict:
                accuracy_dict[id_answer] = []
            accuracy_dict[id_answer].append({"mode": mode, "answer": answer_content})

        if modes[0] == "open" and mode not in ["yes_no_maybe", "options_only"]:
            consistency, mapped_final_answers = calculate_consistency(completions, id2open_answer, mode, dataset_gold)
            print(f"Consistency for mode {mode}: {consistency:.1f}%") 
            for mapped_answer in mapped_final_answers:
                id_answer = mapped_answer['id_question']
                answer_content = mapped_answer['answer']
                if id_answer not in consistency_dict:
                    consistency_dict[id_answer] = []
                consistency_dict[id_answer].append({"mode": mode, "answer": answer_content})
        print("------")
        print()    

    output.append(results)

    result_dir_path = f"out/completions/{model_name}/results"    
    os.makedirs(result_dir_path, exist_ok=True)
    
    with open(f'{result_dir_path}/accuracy_summary.jsonl', 'a') as f:
        for res in output:
            if res["dataset"] == dataset:
                f.write(json.dumps(res) + "\n")
    print("\n") 

    # overall consistency
    combinations = generate_combinations_with_open(modes)
    combinations = [combinations[-1]]
    
    for combo in combinations:
        overall_accuracy = calculate_overall_accuracy(accuracy_dict=accuracy_dict, closed_list=combo)
        overall_results[f"{dataset}_acc"] = round(overall_accuracy, 1)
        print("Modes set:", combo)
        print("Number of combo:", len(combo))
        print("OVERALL ACCURACY:", overall_accuracy)
        overall_accuracy = calculate_overall_accuracy(accuracy_dict=accuracy_dict, only_closed=True, closed_list=combo)
        print("OVERALL ACCURACY (closed only):", overall_accuracy)
        #print("--------")
        print()
        overall_consistency = calculate_overall_consistency(consistency_dict=consistency_dict, closed_list=combo)
        #print("Modes set:", combo)
        #print("Number of combo:", len(combo))
        print("OVERALL CONSISTENCY:", overall_consistency)
        overall_results[f"{dataset}_rel"] = round(overall_consistency, 1)

        overall_consistency = calculate_overall_consistency(consistency_dict=consistency_dict, only_closed=True, closed_list=combo)
        print("OVERALL CONSISTENCY (closed only):", overall_consistency)
        print("--------")
        print()

    

    with open(f"{result_dir_path}/accuracy_{dataset}.json", 'w') as f:
        json.dump(accuracy_dict, f, indent=4)

    with open(f"{result_dir_path}/consistency_{dataset}.json", 'w') as f:
        json.dump(consistency_dict, f, indent=4)

# convert to latex table

import pandas as pd
# load from jsonl
with open(f'{result_dir_path}/accuracy_summary.jsonl', 'r') as f:
    output = [json.loads(line) for line in f.readlines()]

df = pd.DataFrame(output)
# add column avg
# {"model_name": "mediphi", "dataset": "medical_genetics", "open": 92.7,  "mcq": 81.7, "incorrect": 64.6, "roman_numeral": 81.7, "fixed_pos": 78.0, "no_symbols": 67.1, "none_of_the_provided": 42.7}
# Add column 'avg' per modes excluding 'model_name' and 'dataset'
# Transpose the dataframe to have modalities as rows and datasets as columns
df_modality = df.set_index('dataset')[['open', 'mcq', 'incorrect', 'roman_numeral', 'fixed_pos', 'no_symbols', 'none_of_the_provided']].T
# Compute average for MMLU datasets (excluding medqa and medmcqa)
mmlu_datasets = ['professional_medicine', 'college_medicine', 'anatomy', 'clinical_knowledge', 'college_biology', 'medical_genetics']
non_mmlu_datasets = ['medqa', 'medmcqa']

# Calculate mean for each modality across MMLU datasets
df_mmlu = df[df['dataset'].isin(mmlu_datasets)]
df_non_mmlu = df[df['dataset'].isin(non_mmlu_datasets)]

# Average for each modality across MMLU datasets
mmlu_avg = df_mmlu[['open', 'mcq', 'incorrect', 'roman_numeral', 'fixed_pos', 'no_symbols', 'none_of_the_provided']].mean(axis=0)

# Add a new row for MMLU average
df_modality['MMLU'] = mmlu_avg

# Now, compute the final average for each modality:
# (MMLU avg + medqa + medmcqa) / 3
for modality in ['open', 'mcq', 'incorrect', 'roman_numeral', 'fixed_pos', 'no_symbols', 'none_of_the_provided']:
    medqa_val = df_modality.loc[modality, 'medqa'] if 'medqa' in df_modality.columns else float('nan')
    medmcqa_val = df_modality.loc[modality, 'medmcqa'] if 'medmcqa' in df_modality.columns else float('nan')
    mmlu_val = df_modality.loc[modality, 'MMLU']
    vals = [v for v in [medqa_val, medmcqa_val, mmlu_val] if pd.notnull(v)]
    df_modality.loc[modality, 'avg'] = round(sum(vals) / len(vals), 1) if vals else float('nan')
# Add a new column 'avg' with the average across all datasets for each modality
#df_modality['avg'] = df_modality.mean(axis=1).round(1)

# Save the transposed table with averages to LaTeX
df_modality.to_latex(f"{result_dir_path}/accuracy_summary_modality_avg.tex", float_format="%.1f")

cols_to_avg = [col for col in df.columns if col not in ['model_name', 'dataset', 'standard']]
df['avg'] = df[cols_to_avg].mean(axis=1)
df.to_latex(f"{result_dir_path}/accuracy_summary.tex", index=False, float_format="%.1f")
print("\n\n")



# Calculate average accuracy and consistency from overall_results
acc_keys = [k for k in overall_results if k.endswith("_acc")]
rel_keys = [k for k in overall_results if k.endswith("_rel")]

# Calculate MMLU average for accuracy and consistency (reliability)
mmlu_accs = [overall_results[k] for k in acc_keys if any(ds in k for ds in mmlu_datasets)]
mmlu_rels = [overall_results[k] for k in rel_keys if any(ds in k for ds in mmlu_datasets)]

overall_results["mmlu_acc"] = round(sum(mmlu_accs) / len(mmlu_accs), 1) if mmlu_accs else float('nan')
overall_results["mmlu_rel"] = round(sum(mmlu_rels) / len(mmlu_rels), 1) if mmlu_rels else float('nan')

# Now average with medqa and medmcqa
medqa_acc = overall_results.get("medqa_acc", float('nan'))
medmcqa_acc = overall_results.get("medmcqa_acc", float('nan'))
mmlu_acc = overall_results["mmlu_acc"]

medqa_rel = overall_results.get("medqa_rel", float('nan'))
medmcqa_rel = overall_results.get("medmcqa_rel", float('nan'))
mmlu_rel = overall_results["mmlu_rel"]

acc_vals = [v for v in [medqa_acc, medmcqa_acc, mmlu_acc] if pd.notnull(v)]
rel_vals = [v for v in [medqa_rel, medmcqa_rel, mmlu_rel] if pd.notnull(v)]

overall_results["avg_acc_final"] = round(sum(acc_vals) / len(acc_vals), 1) if acc_vals else float('nan')
overall_results["avg_rel_final"] = round(sum(rel_vals) / len(rel_vals), 1) if rel_vals else float('nan')



# if acc_keys:
#     avg_acc = sum(overall_results[k] for k in acc_keys) / len(acc_keys)
#     overall_results["avg_acc"] = round(avg_acc, 1)
# else:
#     avg_acc = 0.0
#     overall_results["avg_acc"] = avg_acc

# if rel_keys:
#     avg_rel = sum(overall_results[k] for k in rel_keys) / len(rel_keys)
#     overall_results["avg_rel"] = round(avg_rel, 1)
# else:
#     avg_rel = 0.0
#     overall_results["avg_rel"] = avg_rel


print(overall_results)
df_overall_results = pd.DataFrame([overall_results])
# Define desired column order: mmlu subsets, mmlu_acc/mmlu_rel, medqa, medmcqa, avg_acc_final, avg_rel_final
mmlu_order = mmlu_datasets + ['mmlu_acc', 'mmlu_rel']
other_order = ['medqa_acc', 'medqa_rel', 'medmcqa_acc', 'medmcqa_rel', 'avg_acc_final', 'avg_rel_final']
ordered_cols = [col for col in mmlu_order + other_order if col in df_overall_results.columns]
# Add any remaining columns (like 'model') at the front
front_cols = [col for col in df_overall_results.columns if col not in ordered_cols]
df_overall_results = df_overall_results[front_cols + ordered_cols]
df_overall_results.to_latex(f"{result_dir_path}/res.tex", index=False, float_format="%.1f")