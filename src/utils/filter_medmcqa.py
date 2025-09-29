import json
from datasets import load_dataset
from collections import Counter

data_all = json.load(open("data/benchmark_open.json"))
data = data_all['medmcqa']

orginal_medmcqa = load_dataset('openlifescienceai/medmcqa', split="validation")
print(orginal_medmcqa[0])
id2subject = {el['id']: el['subject_name'] for el in orginal_medmcqa}

new_data = []
for idx, item in data.items():
    if item['open_question'] != "Not possible":
        item['id'] = idx
        item['subject_name'] = id2subject[idx]
        new_data.append(item)

json.dump(new_data, open("data/medmcqa_open.json", "w"), indent=4)

ds = load_dataset("json", data_files="data/medmcqa_open.json", split="train")
print(len(ds))


# filter samples with no subject name
ds = ds.filter(lambda x: x['subject_name'] is not None and x['subject_name'] != "")
print(len(ds))

subject_counts = Counter(ds['subject_name'])
# ValueError: The least populated class in subject_name column has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
print(subject_counts)
# filter out subjects with less than 2 samples
subjects_to_keep = {subject for subject, count in subject_counts.items() if count >= 2}
ds = ds.filter(lambda x: x['subject_name'] in subjects_to_keep)

print(len(ds))

# sample 1000 items with stratified sampling based on subject
ds = ds.class_encode_column("subject_name")
# get id to subject mapping
id2subject = {i: c for i, c in enumerate(ds.features['subject_name'].names)}
print(id2subject)

# stratified sampling
ds_1000 = ds.train_test_split(test_size=1000, stratify_by_column="subject_name", seed=42)['test']
print(ds_1000)
print(dict(Counter(ds['subject_name'])))
print(dict(Counter(ds_1000['subject_name'])))

ds_1000_ids = list(ds_1000['id'])
# create id to item mapping and remove id from item
id2item_medmcqa = {idx: item for idx, item in data.items() if idx in ds_1000_ids}

data_v2 = {"medqa": data_all['medqa'], "medmcqa": id2item_medmcqa, "mmlu": data_all['mmlu']}

with open("data/benchmark_open_v2.json", "w") as f:
    json.dump(data_v2, f, indent=4)

print("Lengths of datasets:")
print(f"medqa: {len(data_v2['medqa'])}")
print(f"mmlu: {len(data_v2['mmlu'])}")
print(f"medmcqa: {len(data_v2['medmcqa'])}")