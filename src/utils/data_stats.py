import json


subset = "mmlu"
model_name = "medgemma"
input_path = "out/completions/Qwen3-32B-AWQ/judge/mmlu/2025-09-16_07-12-01/judges_mmlu.jsonl" 

with open('data/benchmark.json') as f:
    benchmark = json.load(f)[subset]

with open(f'out/completions/{model_name}/{subset}/generations.jsonl') as f:
    generations = [json.loads(line) for line in f.readlines()]
    id2generation = {el['id_question']: el['completion'] for el in generations}

with open(input_path) as f: 
    data = [json.loads(line) for line in f.readlines()]
    data = [{**el, 'correct_mcq': el['mcq_response'] == el['gold_answer'], 'open_completion': id2generation[el['question_id']], "mcq_options": benchmark[el['question_id']]['options']} for el in data]
print(data[0])
# ['question_id', 'gold_answer', 'mcq_response', 'open_response', 'consistent', 'completion', 'correct', 'justification', 'correct_mcq']

import pandas as pd
df = pd.DataFrame(data)
df = df.drop(columns=['completion'])
df.to_excel(f"{model_name}_{subset}_open_vs_mcq.xlsx", index=False)

accuracy_mcq = sum([el['correct_mcq'] for el in data if el['correct_mcq']]) / len(data) * 100
print("Accuract MCQ:", round(accuracy_mcq, 1))

accuracy_open = sum([el['correct'] for el in data if el['correct']]) / len(data) * 100
print("Accuract Open:", round(accuracy_open, 1))

consistent_responses = [el for el in data if el['consistent']]
inconsistent_responses = [el for el in data if not el['consistent']]
print("Number of consistent responses:", round(len(consistent_responses) / len(data) * 100, 1), "% |", len(consistent_responses))
print("------")
print("Number of NON-consistent responses:", round(len(inconsistent_responses) / len(data) * 100, 1), "% |", len(inconsistent_responses))

inconsistent_responses_open_mcq = [el for el in inconsistent_responses if el['correct'] and not el['correct_mcq']]
print("\twhere Open=correct and MCQ=wrong:", len(inconsistent_responses_open_mcq ))

inconsistent_responses_mcq_open = [el for el in inconsistent_responses if el['correct_mcq'] and not el['correct']]
print("\twhere MCQ=correct and Open=wrong:", len(inconsistent_responses_mcq_open))

inconsistent_responses_mcq_open_false = [el for el in inconsistent_responses if not el['correct_mcq'] and not el['correct']]
print("\twhere MCQ=wrong and Open=wrong (different letters or Not match):", len(inconsistent_responses_mcq_open_false))

inconsistent_responses_mcq_open_correct = [el for el in inconsistent_responses if el['correct_mcq'] and el['correct'] and el['valid_alternative']]
print("\twhere MCQ=correct and Open=Not match (valid alternative: YES):", len(inconsistent_responses_mcq_open_correct))

total_inconsistent = len(inconsistent_responses_open_mcq) + len(inconsistent_responses_mcq_open) + len(inconsistent_responses_mcq_open_false) + len(inconsistent_responses_mcq_open_correct)
print("\tTotal:", total_inconsistent, "| Check:", total_inconsistent == len(inconsistent_responses))

# for el in inconsistent_responses[:10]:
#     print()
#     print("Question ID:", el['question_id'])
#     print("Gold answer:", el['gold_answer'])
#     print("MCQ options:", el['mcq_options'])
#     print("MCQ response:", el['mcq_response'], "| Correct MCQ?", el['correct_mcq'])
#     print("Open response:", el['open_response'], "| Correct Open?", el['correct'])
#     print("Justification:", el['justification'])

if subset == "mmlu":
    print()
    print("---------")
    # result per subset of mmlu (mmlu_anatomy, etc.)
    subsets = set([el['question_id'].split("-")[0] for el in data])
    for sub in subsets:
        data_sub = [el for el in data if el['question_id'].startswith(sub)]
        accuracy_mcq = sum([el['correct_mcq'] for el in data_sub if el['correct_mcq']]) / len(data_sub) * 100
        accuracy_open = sum([el['correct'] for el in data_sub if el['correct']]) / len(data_sub) * 100
        consistent_responses = [el for el in data_sub if el['consistent']]
        inconsistent_responses = [el for el in data_sub if not el['consistent']]
        print(f"Subset: {sub} | Accuract MCQ: {round(accuracy_mcq, 1)} | Accuract Open: {round(accuracy_open, 1)}")
        print("Number of consistent responses:", round(len(consistent_responses) / len(data_sub) * 100, 1), "% |", len(consistent_responses))
        print("Number of NON-consistent responses:", round(len(inconsistent_responses) / len(data_sub) * 100, 1), "% |", len(inconsistent_responses))
        inconsistent_responses_open_mcq = [el for el in inconsistent_responses if el['correct'] and not el['correct_mcq']]
        print("\twhere Open=correct and MCQ=wrong:", len(inconsistent_responses_open_mcq ))
        inconsistent_responses_mcq_open = [el for el in inconsistent_responses if el['correct_mcq'] and not el['correct']]
        print("\twhere MCQ=correct and Open=wrong:", len(inconsistent_responses_mcq_open))
        inconsistent_responses_mcq_open_false = [el for el in inconsistent_responses if not el['correct_mcq'] and not el['correct']]
        print("\twhere MCQ=wrong and Open=wrong (different letters or Not match):", len(inconsistent_responses_mcq_open_false))
        inconsistent_responses_mcq_open_correct = [el for el in inconsistent_responses if el['correct_mcq'] and el['correct'] and el['valid_alternative']]
        print("\twhere MCQ=correct and Open=Not match (valid alternative: YES):", len(inconsistent_responses_mcq_open_correct))
        total_inconsistent = len(inconsistent_responses_open_mcq) + len(inconsistent_responses_mcq_open) + len(inconsistent_responses_mcq_open_false) + len(inconsistent_responses_mcq_open_correct)
        print("\tTotal:", total_inconsistent, "| Check:", total_inconsistent == len(inconsistent_responses))
        print("---------")





