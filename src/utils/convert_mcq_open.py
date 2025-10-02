import json
subset = "mmlu"
model_name =  "Phi-3.5-mini" #"gemini_api/gemini-2.5-flash" #"openai_api/gpt-5-mini"#"together_api/openai/gpt-oss-120b"
input_path = "out/completions/Qwen3-32B-AWQ/judge/Phi-3.5-mini/mmlu/2025-09-19_17-41-46/judges_mmlu.jsonl" 


with open(input_path) as f: 
    data = [json.loads(line) for line in f.readlines()]

    for mode in ["mcq", "open"]:
        with open(f'out/completions/{model_name}/{subset}/generations_{mode}.jsonl', 'w') as f:
            for item in data:
                out_dict = {
                    "id_question": item["question_id"],
                    "mode": mode,
                    "gold_answer": item['gold_answer'],
                    "final_answer": item[f'{mode}_response'],
                    "correct": item['gold_answer'] == item[f'{mode}_response']
                }
                if mode == "open":
                    out_dict['valid_alternative'] = item['valid_alternative']

                json.dump(out_dict, f)
                f.write("\n")


# {"question_id": "0013", "gold_answer": "D", "mcq_response": "D", "open_response": "D", "consistent": true, "completion": "\n\nMapped Option: D  \nJustification: When the patient stands, venous return decreases, leading to a lower left ventricular end-diastolic volume. This reduces stroke volume and shortens diastolic filling time due to increased heart rate. The murmur intensity decreases, which aligns with the hemodynamic change of reduced diastolic filling time (Option D). Other options (A, B, C) are inconsistent with the expected physiological response to standing.", "correct": true, "justification": "When the patient stands, venous return decreases, leading to a lower left ventricular end-diastolic volume. This reduces stroke volume and shortens diastolic filling time due to increased heart rate. The murmur intensity decreases, which aligns with the hemodynamic change of reduced diastolic filling time (Option D). Other options (A, B, C) are inconsistent with the expected physiological response to standing.", "valid_alternative": null, "justification_alternative": null}

# exepected
# {"id_question": "professional_medicine-270", "mode": "fixed_pos", "gold_answer": "D", "final_answer": "D", "correct": true, "completion": "\\boxed{D}", "thinking": "We need to decide best management of a swallowed nail (4 cm) located in left upper quadrant, presumably stomach region. Patient asymptomatic, no perforation. The nail is sharp? 4-cm nail likely pointed. Sharp objects >2 cm usually need removal. Guidelines: Sharp objects in stomach should be removed endoscopically if reachable within 24 hours. If beyond stomach, observation maybe. But location left upper quadrant could be stomach. Endoscopic removal is recommended. So answer D.\n\nCheck other options: ipecac is contraindicated for sharp objects. Observation not recommended for sharp objects >2 cm. Open laparotomy is invasive not needed. So D.\n\nThus answer: \\boxed{D}.", "thinking_length": 158}
