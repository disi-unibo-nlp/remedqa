import json
import itertools
import os
import argparse

possible_ids_medqa = json.load(open("data/processed/benchmark_open_filtered.json"))['medqa'].keys()
possible_ids_medmcqa = json.load(open("data/processed/benchmark_open_filtered.json"))['medmcqa'].keys() 
possible_ids_mmlu = json.load(open("data/processed/benchmark_open_filtered.json"))['mmlu'].keys()
possible_ids_mmlu_dict = {}

for key in possible_ids_mmlu:
    subset = key.split("-", 1)[0]
    if subset not in possible_ids_mmlu_dict:
        possible_ids_mmlu_dict[subset] = []
    possible_ids_mmlu_dict[subset].append(key)

def generate_combinations(items, required="open"):
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
    correct_ids = []
    for key, answers in accuracy_dict.items():
        if only_closed:
            all_answers = [el['answer'] for el in answers if el['mode'] != "open" and el['mode'] in list(closed_list)]
        else:
            all_answers = [el['answer'] for el in answers if el['mode'] in list(closed_list)]
        print("All answer:" , all_answers)
               
        if sum(all_answers) == len(all_answers):
            correct_ids.append(key)
            consistent.append(True)
        else:
            consistent.append(False)

   
    total = len(consistent)
    overall_accuracy = sum(consistent) / total * 100 if total > 0 else 0
    return overall_accuracy, correct_ids

def calculate_overall_consistency(consistency_dict_correct, consistency_dict_wrong, only_closed=False, closed_list=None):
    consistent = [1] * len(consistency_dict_correct)
    from collections import Counter
    fails = []
    for key, answers in consistency_dict_wrong.items():
        if only_closed:
            all_answers = [el['answer'] for el in answers if el['answer'] and el['mode'] != "open" and el['mode'] in list(closed_list)]
        else:
            all_answers = [el['answer'] for el in answers if el['answer'] and el['mode'] in list(closed_list)]
        
        try: 
            answer_dict = dict(Counter(all_answers)) if all_answers else {}
            print("Answer dict:", answer_dict)
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

def normalize_answers(permutation_completions, reference_completions_dict, mode, dataset_gold):
    """
    Normalize answers from different formats to a standard format.
    
    Args:
        permutation_completions: List of completions to normalize
        reference_completions_dict: Dictionary of reference completions
        mode: The format mode (e.g., 'roman_numeral', 'incorrect', 'fixed_pos', etc.)
        dataset_gold: Gold standard dataset containing original question data
    
    Returns:
        mapped_final_answers: List of dicts with normalized answers
    """
    mapped_final_answers = []
    
    for completion in permutation_completions:
        id_completion = completion['id_question']
        final_answer_permutation = completion['final_answer']
        
        # Skip if no reference or no answer
        if id_completion not in reference_completions_dict or not final_answer_permutation:
            continue
        
        # Normalize based on mode
        if mode in ["roman_numeral", "options_only_roman_numeral"]:
            roman2letter = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D'}
            normalized_answer = roman2letter.get(final_answer_permutation, final_answer_permutation)
            mapped_final_answers.append({"id_question": id_completion, "answer": str(normalized_answer)})
            print("Mapped roman numeral:", normalized_answer)
            
        elif mode in ["incorrect", "options_only_incorrect"]:
            # For incorrect mode, convert list of 3 incorrect answers to the 1 correct answer
            if final_answer_permutation and len(final_answer_permutation) == 3:
                final_answer_incorrect = set(['A', 'B', 'C', 'D']).difference(set(final_answer_permutation))
                normalized_answer = next(iter(final_answer_incorrect))
                mapped_final_answers.append({"id_question": id_completion, "answer": str(normalized_answer)})
            else:
                # Invalid format - keep as is
                mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_permutation)})
            
        elif mode in ["fixed_pos", "options_only_fixed_pos"]:
            # Map answer from fixed position format to original format
            dataset_fixed_pos = dataset_gold[mode]
            id2option_fixed_pos = dataset_fixed_pos[id_completion]['options']
            
            # Get the option text from fixed position format
            mapped_answer_permutation = id2option_fixed_pos.get(final_answer_permutation, "")
            mapped_answer_permutation = mapped_answer_permutation.replace(".", "").replace("[", "").replace("]", "")
            
            # Get original dataset mapping
            dataset_open = dataset_gold.get('mcq', dataset_gold.get('options_only_mcq'))
            id2option_open = dataset_open[id_completion]['options']
            option2id_open = {v.replace(".", "").replace("[", "").replace("]", ""): k 
                            for k, v in id2option_open.items()}
            
            # Map to original answer ID
            normalized_answer = option2id_open.get(mapped_answer_permutation, "")
            mapped_final_answers.append({"id_question": id_completion, "answer": str(normalized_answer)})
            print("Mapped fixed pos:", normalized_answer)
            
        elif mode in ["no_symbols", "options_only_no_symbols"]:
            # Remove symbols and map back to original format
            final_answer_permutation = final_answer_permutation.replace(".", "").replace("[", "").replace("]", "")
            
            dataset_open = dataset_gold.get('mcq', dataset_gold.get('options_only_mcq'))
            id2option_open = dataset_open[id_completion]['options']
            option2id_open = {v.replace(".", "").replace("[", "").replace("]", ""): k 
                            for k, v in id2option_open.items()}
            
            normalized_answer = option2id_open.get(final_answer_permutation, "")
            mapped_final_answers.append({"id_question": id_completion, "answer": str(normalized_answer)})
            print("Mapped no symbols:", normalized_answer)
            
        else:
            # Default: no normalization needed
            mapped_final_answers.append({"id_question": id_completion, "answer": str(final_answer_permutation)})
            print("Mapped default:", final_answer_permutation)
    
    return mapped_final_answers


def calculate_consistency(mapped_final_answers, reference_completions_dict, mode, dataset_gold):
    """
    Calculate consistency between normalized answers and reference answers.
    
    Args:
        mapped_final_answers: List of normalized answers from normalize_answers()
        reference_completions_dict: Dictionary of reference completions
        mode: The format mode (used for special comparison logic)
        dataset_gold: Gold standard dataset (used for certain modes)
    
    Returns:
        consistency: Percentage of consistent answers
    """
    consistent = []
    
    for mapped_answer in mapped_final_answers:
        id_completion = mapped_answer['id_question']
        normalized_answer = mapped_answer['answer']
        
        # Get reference answer
        if id_completion not in reference_completions_dict:
            continue
            
        reference_completion = reference_completions_dict[id_completion]
        final_answer_reference = reference_completion['final_answer']
        
        # Compare based on mode
        if mode in ["incorrect", "options_only_incorrect"]:
            # For incorrect mode, the original answer was a list of 3 incorrect options
            # We need to check if the reference is NOT in that original list
            # But at this point, we've already normalized to the correct answer
            # So we just compare normally
            if normalized_answer:
                # Check if we have the original permutation to validate
                # Since we only have normalized answer, we compare directly
                consistent.append(final_answer_reference == normalized_answer)
            else:
                consistent.append(False)
                
        elif mode in ["fixed_pos", "options_only_fixed_pos", "no_symbols", "options_only_no_symbols"]:
            # For these modes, we need to compare the normalized answer to reference
            dataset_open = dataset_gold.get('mcq', dataset_gold.get('options_only_mcq'))
            id2option_open = dataset_open[id_completion]['options']
            
            if not isinstance(final_answer_reference, list) and final_answer_reference in id2option_open:
                consistent.append(final_answer_reference == normalized_answer)
            else:
                consistent.append(False)
                
        else:
            # Default comparison
            print("Comparing:", final_answer_reference, "vs", normalized_answer)
            consistent.append(final_answer_reference == normalized_answer)
    
    total = len(consistent)
    consistency = sum(consistent) / total * 100 if total > 0 else 0
    
    return consistency


def calculate_accuracy(completions, mode):
   
    if mode in ["roman_numeral", "options_only_roman_numeral"]:
        roman2letter = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D'}
        if completions[0]['final_answer'] in roman2letter and not completions[0]['gold_answer'] in roman2letter:
            correct = [{'id_question': el['id_question'], 'answer': roman2letter[el['gold_answer']] == el['final_answer']} for el in completions]
        else:
            correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'] == el['final_answer']} for el in completions]
    elif mode in ["incorrect", "options_only", "options_only_incorrect"]:
        correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'] == el['final_answer']} for el in completions]
    elif mode == "open":
        correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'].replace(".","").replace("[","").replace("]","") == el['final_answer'].replace(".","").replace("[","").replace("]","") or (el['valid_alternative'] == "Yes") if el['final_answer'] and el['gold_answer'] and not isinstance(el['final_answer'], list) else False} for el in completions]
    else:
        correct = [{'id_question': el['id_question'], 'answer': el['gold_answer'].replace(".","") == el['final_answer'].replace(".","") if el['final_answer'] and el['gold_answer'] else False} for el in completions]

    correct_list = [el['answer'] for el in correct]
    total = len(correct_list)
    accuracy = sum(correct_list) / total * 100 if total > 0 else 0
    return accuracy, correct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and modes.")
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=["medqa", "medmcqa", 'professional_medicine', 'college_medicine', 'anatomy', 'clinical_knowledge', 'college_biology', 'medical_genetics'],
        help="Dataset subset to use (mmlu, medqa, or medmcqa)."
    )

    parser.add_argument(
        "--options-only-mode",
        action="store_true",
        help="If set, restricts evaluation to option-only modes (overrides --modes)."
    )

    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=[
            "open", 
            "mcq", 
            "incorrect", 
            "none_of_the_provided", 
            "roman_numeral", 
            "fixed_pos", 
            "no_symbols",
            "options_only_mcq", 
            "options_only_incorrect", 
            "options_only_roman_numeral", 
            "options_only_none_of_the_provided", 
            "options_only_fixed_pos", 
            "options_only_no_symbols"
        ],
        default=None,
        help=(
            "List of modes to evaluate. "
            "If not provided, defaults are determined by --option_only_enabled. "
            "Examples: --modes open mcq incorrect"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="together_api/openai/gpt-oss-120b",
        help="Name or path of the model to evaluate."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions",
        help="Output directory to save results."
    )

    parser.add_argument(
        "--reference-mode",
        type=str,
        choices=[
            "open", 
            "mcq", 
            "incorrect", 
            "none_of_the_provided", 
            "roman_numeral", 
            "fixed_pos", 
            "no_symbols",
            "options_only_mcq", 
            "options_only_incorrect", 
            "options_only_roman_numeral", 
            "options_only_none_of_the_provided", 
            "options_only_fixed_pos", 
            "options_only_no_symbols"
        ],
        default="open",
        help="Reference mode for consistency calculations."
    )

    args = parser.parse_args()


    options_only_enabled = args.options_only_mode
    reference_mode = args.reference_mode

    if args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = ["medqa" , "medmcqa",'professional_medicine', 'college_medicine', 'anatomy', 'clinical_knowledge', 'college_biology', 'medical_genetics']
    
    if args.modes is not None:
        modes = args.modes
        # ensure 'open' is always as first mode
        if "open" in modes:
            modes.remove("open")
            modes = ["open"] + modes
    else:
        modes = ["open", "mcq", "incorrect", "roman_numeral", "fixed_pos", "no_symbols","none_of_the_provided"] 

    if options_only_enabled:
        modes = ['options_only_' + mode for mode in modes if mode != "open"]
        # ensure 'options_only_mcq' is first 
        if "options_only_mcq" in modes:
            modes.remove("options_only_mcq")
            modes = ["options_only_mcq"] + modes
            
    model_name = args.model_name #"together_api/openai/gpt-oss-120b" #"openai_api/gpt-5-mini" # #"openai_api/gpt-5-mini" #"together_api/openai/gpt-oss-120b" #"openai_api/gpt-5-mini"  # "llama3"#"gemini_api/gemini-2.5-flash" #"together_api/meta-llama/Llama-3.3-70B-Instruct-Turbo" # #"llama3" #"openai_api/gpt-5-mini" ##"llama3" # # # #"medgemma" # # # #"openai_api/gpt-5-mini" #  # # #"openai_api/gpt-5-mini" #"gemini_api/gemini-2.5-flash" # # # #

    output = []
    id2reference_answer = {}

    overall_results = {"model": model_name.split("/")[-1]}
    for dataset in datasets:
        
        dataset_dir_name = dataset if dataset in ["medqa", "medmcqa"] else "mmlu"
        
        if not options_only_enabled:
            with open(f"data/processed/{dataset_dir_name}_all.json") as f:
                dataset_gold = json.load(f)
        else:
            with open(f"data/processed/{dataset_dir_name}_options_only.json") as f:
                dataset_gold = json.load(f)

        if options_only_enabled:
            dataset_dir_name = dataset_dir_name + "_options_only"
        
        consistency_dict = {}
        accuracy_dict = {}
        results = {}
        results["model_name"] = model_name
        results["dataset"] = dataset
        
        for mode in modes:
            
            print(f"Calculating accuracy for mode: {mode} | dataset: {dataset}")
            with open(f'out/completions/{model_name}/{dataset_dir_name}/generations_{mode}.jsonl', 'r') as f:
                completions = [json.loads(line) for line in f.readlines()]
            completions = filter_completions(completions, dataset)

            if mode in ["open", "options_only_mcq"]:
                id2reference_answer = {item['id_question']: item for item in completions}
            
            accuracy, final_answers = calculate_accuracy(completions, mode)
            
            results[mode] = round(accuracy, 1)
            
            print(f"Accuracy for mode {mode}: {accuracy:.1f}%")

            for answer in final_answers:
                id_answer = answer['id_question']
                answer_content = answer['answer']
                if id_answer not in accuracy_dict:
                    accuracy_dict[id_answer] = []
                accuracy_dict[id_answer].append({"mode": mode, "answer": answer_content})
                
            # First, normalize the answers
            mapped_final_answers = normalize_answers(
                completions, 
                id2reference_answer, 
                mode, 
                dataset_gold
            )

            # Store normalized answers for overall consistency calculation
            for mapped_answer in mapped_final_answers:
                id_answer = mapped_answer['id_question']
                answer_content = mapped_answer['answer']
                if id_answer not in consistency_dict:
                    consistency_dict[id_answer] = []
                consistency_dict[id_answer].append({"mode": mode, "answer": answer_content})

            # print consistency dict
            print("Consistency dict:", consistency_dict)
            
            print("------")
            print()    

        output.append(results)

        result_dir_path = f"out/completions/{model_name}/results" if not options_only_enabled else f"out/completions/{model_name}/results_options_only"
        os.makedirs(result_dir_path, exist_ok=True)
        
        with open(f'{result_dir_path}/accuracy_summary.jsonl', 'a') as f:
            for res in output:
                if res["dataset"] == dataset:
                    f.write(json.dumps(res) + "\n")
        print("\n") 

        
        overall_accuracy, correct_ids = calculate_overall_accuracy(accuracy_dict=accuracy_dict, closed_list=modes)
        overall_results[f"{dataset}_acc"] = round(overall_accuracy, 1)
        print("Modes set:", modes)
        print("Number of permutations:", len(modes))
        print("OVERALL ACCURACY:", overall_accuracy)
       
        print()
        # filter consistency_dict to only include correct ids
        consistency_dict_correct = {k: v for k, v in consistency_dict.items() if k in correct_ids}
        consistency_dict_wrong = {k: v for k, v in consistency_dict.items() if k not in correct_ids}

        overall_consistency = calculate_overall_consistency(consistency_dict_correct=consistency_dict_correct, consistency_dict_wrong=consistency_dict_wrong, closed_list=modes)
        print("OVERALL CONSISTENCY:", overall_consistency)
        overall_results[f"{dataset}_rel"] = round(overall_consistency, 1)

        for reference_mode in modes:
            ref_mode_dict = {"ref_mode": reference_mode, "model_name": model_name.split("/")[-1], "dataset": dataset, "results": {}}
            
            non_reference_modes = [mode for mode in modes if mode != reference_mode]
            for non_ref_mode in non_reference_modes:
                consistency_per_mode = calculate_overall_consistency(consistency_dict_correct=consistency_dict_correct, consistency_dict_wrong=consistency_dict_wrong, closed_list=[reference_mode, non_ref_mode])
                print(f"OVERALL CONSISTENCY ({non_ref_mode} vs {reference_mode}):", consistency_per_mode)
                ref_mode_dict["results"][non_ref_mode] = round(consistency_per_mode, 1)
            
            with open(f"{result_dir_path}/consistency_vs_reference.jsonl", 'a') as f:
                json.dump(ref_mode_dict, f)
                f.write("\n")

        print("--------")
        print()

        with open(f"{result_dir_path}/accuracy_{dataset}.json", 'w') as f:
            json.dump(accuracy_dict, f, indent=4)

        with open(f"{result_dir_path}/consistency_{dataset}.json", 'w') as f:
            json.dump(consistency_dict, f, indent=4)

    if len(set(datasets)) == 8:
    
        # convert to latex table
        import pandas as pd
        # load from jsonl
        with open(f'{result_dir_path}/accuracy_summary.jsonl', 'r') as f:
            output = [json.loads(line) for line in f.readlines()]

        df = pd.DataFrame(output)
        
        modes_df = modes 
        # Transpose the dataframe to have modalities as rows and datasets as columns
        df_modality = df.set_index('dataset')[modes_df].T
        # Compute average for MMLU datasets (excluding medqa and medmcqa)
        mmlu_datasets = ['professional_medicine', 'college_medicine', 'anatomy', 'clinical_knowledge', 'college_biology', 'medical_genetics']
        non_mmlu_datasets = ['medqa', 'medmcqa']

        # Calculate mean for each modality across MMLU datasets
        df_mmlu = df[df['dataset'].isin(mmlu_datasets)]
        df_non_mmlu = df[df['dataset'].isin(non_mmlu_datasets)]

        # Average for each modality across MMLU datasets
        mmlu_avg = df_mmlu[modes_df].mean(axis=0)

        # Add a new row for MMLU average
        df_modality['MMLU'] = mmlu_avg

        # Now, compute the final average for each modality:
        # (MMLU avg + medqa + medmcqa) / 3
        for modality in modes_df:
            medqa_val = df_modality.loc[modality, 'medqa'] if 'medqa' in df_modality.columns else float('nan')
            medmcqa_val = df_modality.loc[modality, 'medmcqa'] if 'medmcqa' in df_modality.columns else float('nan')
            mmlu_val = df_modality.loc[modality, 'MMLU']
            vals = [v for v in [medqa_val, medmcqa_val, mmlu_val] if pd.notnull(v)]
            df_modality.loc[modality, 'avg'] = round(sum(vals) / len(vals), 1) if vals else float('nan')
        
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