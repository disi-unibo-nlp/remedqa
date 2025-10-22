import json
import pandas as pd

dataset_name = "medmcqa"
batch_out_file_path = "out/openai_batch/20250909_100941/annotations.json"

with open('data/benchmark.json', 'r') as f:
    benchmark_data = json.load(f)[dataset_name]
    # "0018": {
    #         "question": "A 29-year-old man presents to the emergency department due to central chest pain over the past 3 days which is constant and unrelated to exertion. The pain is sharp, severe, increases when lying down, and improves with leaning forward. The pain also radiates to his shoulders and neck. The patient has no past medical history. He has smoked 10 cigarettes per day for the past 7 years and occasionally drinks alcohol. He presents with vital signs: blood pressure 110/70 mm Hg, regular radial pulse of 95/min, and temperature 37.3\u00b0C (99.1\u00b0F). On physical exam, a scratching sound of to-and-from character is audible over the left sternal border at end-expiration with the patient leaning forward. His chest X-ray is normal and ECG is shown in the picture. Which of the following is the optimal therapy for this patient?",
    #         "options": {
    #             "A": "Indomethacin +/- omeprazole",
    #             "B": "Ibuprofen + colchicine +/- omeprazole",
    #             "C": "Pericardiocentesis",
    #             "D": "Pericardiectomy"
    #         },
    #         "answer": "B"
    #     },
    #benchmark_data = [ {'id': k, **v} for k, v in benchmark_data.items()]

benchmark_df = pd.DataFrame(benchmark_data)

with open(batch_out_file_path, 'r') as f:
    annotations = json.load(f)
    #annotations_df = pd.DataFrame(annotations)
    #  "0001": {
    #     "original": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
    #     "rewritten": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which action?",
    #     "rationale": "The question asks for a single most likely mechanism of action for the drug causing the patient's symptoms, which can be directly mapped to one of the options.",
    #     "safe": true,
    #     "changed_part": "The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
    #     "new_part": "The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which action?"
    # },

keys = list(annotations.keys())

results = {}
for k in keys:
    item = benchmark_data[k]
    ann = annotations[k]
    results[k] = { 
        "original_question": item['question'],
        "open_question": ann['rewritten'],
        "is_converted": ann['safe'],
        "explanation": ann['rationale'],
        "replaced_part": ann['changed_part'],
        "new_part": ann['new_part'],
        "options": item['options'],
        "correct_option": item['answer'],
        "open_answer":  item['options'][item['answer']]
    }

# save results to json
with open(f'data/{dataset_name}_open.json', 'w') as f:
    json.dump(results, f, indent=4)
