import json
final_dict = {}
for dataset_name in ["medqa", "medmcqa", "mmlu"]:
    with open(f'data/{dataset_name}_open.json', 'r') as f:
        data = json.load(f)
    
    final_dict[dataset_name] = data

with open('data/benchmark_open.json', 'w') as f:
    json.dump(final_dict, f, indent=4)