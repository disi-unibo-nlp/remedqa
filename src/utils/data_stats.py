import json

with open('data/benchmark.json') as f:
    benchmark = json.load(f)['medqa']

with open('out/completions/medgemma/medqa/generations.jsonl') as f:
    generations = [json.loads(line) for line in f.readlines()]
    id2generation = {el['id_question']: el['completion'] for el in generations}

with open('out/completions/Qwen3-32B-AWQ/judge/medqa/2025-09-10_10-51-32/judges.jsonl') as f: 
    data = [json.loads(line) for line in f.readlines()]
    data = [{**el, 'correct_mcq': el['mcq_response'] == el['gold_answer'], 'open_completion': id2generation[el['question_id']], "mcq_options": benchmark[el['question_id']]['options']} for el in data]
print(data[0])
# ['question_id', 'gold_answer', 'mcq_response', 'open_response', 'consistent', 'completion', 'correct', 'justification', 'correct_mcq']

import pandas as pd
df = pd.DataFrame(data)
df = df.drop(columns=['completion'])
df.to_excel("medgemma_medqa_open_vs_mcq.xlsx", index=False)

accuracy_mcq = sum([el['correct_mcq'] for el in data if el['correct_mcq']]) / len(data) * 100
print("Accuract MCQ:", round(accuracy_mcq, 1))

accuracy_open = sum([el['correct'] for el in data if el['correct']]) / len(data) * 100
print("Accuract Open:", round(accuracy_open, 1))

consistent_responses = [el for el in data if el['consistent']]
inconsistent_responses = [el for el in data if not el['consistent']]
print("Number of consistent responses:", round(len(consistent_responses) / len(data) * 100, 1), "% |", len(consistent_responses))
print("Number of NON-consistent responses:", round(len(inconsistent_responses) / len(data) * 100, 1), "% |", len(inconsistent_responses))

inconsistent_responses_open_mcq = [el for el in inconsistent_responses if el['correct']]
print("Number of cases where Open=correct and MCQ=wrong:", len(inconsistent_responses_open_mcq ))


inconsistent_responses_mcq_open = [el for el in inconsistent_responses if el['correct_mcq']]
print("Number of cases where MCQ=correct and Open=wrong:", len(inconsistent_responses_mcq_open))





