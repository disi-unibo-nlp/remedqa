import json
from collections import Counter

benchmark_closed = json.load(open("data/benchmark.json"))
benchmark_open = json.load(open("data/benchmark_open_v2.json"))

# filter out open benchmark for "open_question": "Not possible"
benchmark_open_filtered = {}

for dataset, items in benchmark_open.items():
   
    filtered_items = {k: v for k, v in items.items() if v.get("open_question", "") != "Not possible"}
    benchmark_open_filtered[dataset] = filtered_items

print("Lengths of filtered datasets:")
print(f"medqa: {len(benchmark_open_filtered['medqa'])} / {len(benchmark_closed['medqa'])}")
print(f"medmcqa: {len(benchmark_open_filtered['medmcqa'])} / {len(  benchmark_closed['medmcqa'])}")

print(f"mmlu: {len(benchmark_open_filtered['mmlu'])} / {len(benchmark_closed['mmlu'])}")
# print stats mmlu per subset given by key.split("-")[0]
mmlu_subset_counts = dict(Counter([k.split("-")[0] for k,v in benchmark_open_filtered['mmlu'].items()]))
print("MMLU subset counts in open benchmark:")
for subset, count in mmlu_subset_counts.items():
    print(f"{subset}: {count} / {len([k for k in benchmark_closed['mmlu'].keys() if k.startswith(subset)])}")





ids_medqa = set(benchmark_closed['medqa'].keys())
ids_mmlu = set(benchmark_closed['mmlu'].keys())
ids_medmcqa = set(benchmark_closed['medmcqa'].keys())

# filter closed benchmark to only include ids present in open benchmark
benchmark_closed_filtered = {
    "medqa": {k: v for k, v in benchmark_closed['medqa'].items() if k in benchmark_open_filtered['medqa']},
    "mmlu": {k: v for k, v in benchmark_closed['mmlu'].items() if k in benchmark_open_filtered['mmlu']},
    "medmcqa": {k: v for k, v in benchmark_closed['medmcqa'].items() if k in benchmark_open_filtered['medmcqa']},
}

# add id key to each item in mcq datasets
for dataset in ["medqa", "medmcqa", "mmlu"]:
    for k, v in benchmark_closed_filtered[dataset].items():
        v['id'] = k
        
print()
print("Lengths of closed datasets after filtering:")
print(f"medqa: {len(benchmark_closed_filtered['medqa'])} / {len(benchmark_closed['medqa'])}")
print(f"medmcqa: {len(benchmark_closed_filtered['medmcqa'])} / {len(benchmark_closed['medmcqa'])}")
print(f"mmlu: {len(benchmark_closed_filtered['mmlu'])} / {len(benchmark_closed['mmlu'])}")

# do sanity check to ensure all ids in closed filtered are in open filtered
assert set(benchmark_closed_filtered['medqa'].keys()).issubset(set(benchmark_open_filtered['medqa'].keys()))
assert set(benchmark_closed_filtered['medmcqa'].keys()).issubset(set(benchmark_open_filtered['medmcqa'].keys()))
assert set(benchmark_closed_filtered['mmlu'].keys()).issubset(set(benchmark_open_filtered['mmlu'].keys()))

# check lengths equal
assert len(benchmark_closed_filtered['medqa']) == len(benchmark_open_filtered['medqa'])
assert len(benchmark_closed_filtered['medmcqa']) == len(benchmark_open_filtered['medmcqa'])
assert len(benchmark_closed_filtered['mmlu']) == len(benchmark_open_filtered['mmlu'])


# save filtered datasets
import os
os.makedirs("data/processed", exist_ok=True)


with open("data/processed/benchmark_closed_filtered.json", "w") as f:
    json.dump(benchmark_closed_filtered, f, indent=4)

print("MCQ datasets saved to data/processed/benchmark_closed_filtered.json")