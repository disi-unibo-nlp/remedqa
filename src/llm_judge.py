from huggingface_hub import login
import os
import json 
import time
import logging
import argparse
import re
import numpy as np

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
from openai import OpenAI

from datetime import datetime
# Load variables from the .env file
load_dotenv()
np.random.seed(42)
# m42-health/Llama3-Med42-8B  meta-llama/Llama-3.1-8B-Instruct
# Manually set the required environment variable
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-32B-AWQ", help="Model's HF directory or local path")
    parser.add_argument("--dataset_path", type=str, default="data/benchmark.json", help="Dataset HF directory")
    parser.add_argument("--out_dir", type=str, default="./out", help="Outputs directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of data to process in train set. Default is -1 to process all data.")
    parser.add_argument("--start_idx", type=int, default=0, help="Index of first prompt to process.")
    parser.add_argument("--batch_size", type=int, default=12, help="Maximum number of data to process per batch.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory to store model weights")
    parser.add_argument("--max_model_len", type=int, default=8000, help="Maximum input sequence length")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p sampling.")
    parser.add_argument("--n_out_sequences", type=int, default=1, help="Number of generated sequences per instance")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature parameter")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use for inference.")
    parser.add_argument("--max_tokens", type=int, default=32000, help="Max number of tokens to generate in CoT prompting.")
    parser.add_argument("--subset", type=str, default="medmcqa", help="Subset of the dataset to use")
    parser.add_argument("--mode", type=str, choices=["judge"], default="judge", help="Mode of operation: single or multiple attempts")
    parser.add_argument("--strategy", type=str, choices=["cot", "direct"], default="cot", help="Inference strategy: chain of thought (cot) or direct")


    return parser.parse_args()

# def extract_answer(response):   
#     # For standard mode, we expect a single answer
#     match = re.search(r'### Answer:\s*([A-D])', response)
#     if match:
#         return match.group(1)
#     else:
#         return "E"

import re

def extract_json_block(text, parse=True):
    """
    Extracts the content between ```json and ``` in a string.
    
    Args:
        text (str): Input string containing a ```json ... ``` block.
        parse (bool): If True, return parsed JSON object (dict/list).
                      If False, return raw JSON string.

    Returns:
        dict/list/str or None: Parsed JSON (default) or raw string. 
                               Returns None if no block found or JSON invalid.
    """
    # regex to capture content between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return None  # no JSON block found

    json_str = match.group(1).strip()

    if parse:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None  # invalid JSON
    else:
        return json_str

def extract_answer(response: str):
    """
    Extracts the mapped option(s) from an LLM judge response.
    
    Expected format:
    Mapped Option: A
    Justification: ...
    
    Handles:
    - Single answers (e.g., "A")
    - Multiple answers (e.g., "A or B", "A, B")
    - 'No clear match' case
    """
    match = re.search(r"Mapped Option:\s*(.+)", response, re.IGNORECASE)
    if not match:
        return None  # nothing found
    
    raw_answer = match.group(1).strip()
    
    # Normalize cases like "A or B" / "A, B"
    if raw_answer.lower().startswith("no clear match"):
        return "No clear match"
    
    options = re.split(r"\s*(?:,|or|and)\s*", raw_answer, flags=re.IGNORECASE)
    options = [opt.strip().upper() for opt in options if opt.strip()]
    
    # Return single answer as str, multiple as list
    if len(options) == 1:
        return options[0]
    return options

def extract_mcq_answer(response: str):
    """
    Extracts the MCQ letter from a response containing 'Final Answer: (X)'.
    
    Handles cases like:
    - 'Final Answer: (A)'
    - 'Final Answer: A'
    - 'Answer: (B)'
    - 'Final Answer: C'
    Returns the option letter as a string (e.g., 'A'), or None if not found.
    """
    match = re.search(r"(?:Final Answer:|Answer:)\s*\(?([A-D])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None



if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token="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")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")

    

    # parse input args
    args = parse_arguments()
    print(args)

    actual_model_name = args.model_path.split("/")[-1]
    output_dir = f"./out/completions/{actual_model_name}/{args.mode}/{args.subset}/" + now_dir
    os.makedirs(output_dir, exist_ok=True)

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"{output_dir}/output.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Output dir set to: {output_dir}")



    if args.n_gpus > 1: 
        import ray
        ray.init(_temp_dir="/my_local_tmp_dir", log_to_driver=False)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    sampling_params = SamplingParams(
        n=args.n_out_sequences, 
        temperature=0.6, 
        top_p=0.95, 
        top_k=20,
        max_tokens=args.max_tokens, 
        logprobs=5,
        seed=0
    )

    
    llm = LLM(
        model=args.model_path,
        #tokenizer=args.model_path,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_path.lower() else "auto",
        quantization="awq" if "awq" in args.model_path.lower() else None,
        enforce_eager=True,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        trust_remote_code=True,
        tensor_parallel_size=args.n_gpus
    )

    with open(args.dataset_path.replace(".json", "_open.json")) as f:
        dataset_open = json.load(f)[args.subset]

    dataset_open = [(idx, item) for (idx, item) in dataset_open.items()]
    id_open2question = {idx: item['open_question'] for (idx, item) in dataset_open}
    non_possible_ids = [idx for (idx, item) in dataset_open if item['open_question'] == 'Not possible']
    #dataset_open = [(idx, item) for (idx, item) in dataset_open if el['open_question'] != 'Not possible']
    

    with open(f'out/completions/medgemma/{args.subset}/generations.jsonl') as f:
        completions = [json.loads(line) for line in f.readlines()]
        completions = [el for el in completions if el['id_question'] not in non_possible_ids]
        completion_mcq = completions[::2]
        completions = completions[1::2]
        completions_ids = [item["id_question"] for item in completions]

    with open(args.dataset_path) as f:
        dataset = json.load(f)[args.subset]
    dataset = [(idx, item) for idx, item in dataset.items() if idx in completions_ids]

    id2answer_mcq = {}
    for compl in completion_mcq:
        compl_id = compl['id_question']
        compl = compl['completion'].split("Final Answer:")[1].strip() if "Final Answer:" in compl['completion'] else "None"
        extracted_answer = extract_mcq_answer("Final Answer: " + compl)
        #print(extracted_answer)
        id2answer_mcq[compl_id] = extracted_answer
    
    id2answer_open = {}
    for compl in completions:
        compl_id = compl['id_question']
        compl = compl['completion'].strip() if "Final Answer:" in compl['completion'] else "None"
        extracted_answer = compl
        id2answer_open[compl_id] = extracted_answer

    
    from collections import Counter
    data_dict = dict(Counter(completions_ids))
    sorted_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))
    print(sorted_dict)
    assert len(completion_mcq) == len(completions)
    print("LENGTH COMPLETION",len(completions) )
    
    print("LENGTH DATASET",len(dataset) )
    
    #completions = completions_open
    assert len(dataset) == len(completions)

    # Adjust dataset based on start_idx
    if args.start_idx:
        assert args.start_idx < len(dataset), "start_idx is greater than the dataset size."
        dataset = dataset[args.start_idx:]

    # Adjust dataset based on max_samples
    if args.max_samples != -1:
        assert args.max_samples > 0, "max_samples should be greater than 0."
        dataset = dataset[:args.max_samples]

    logger.info(f"Processed {len(dataset)} records from {args.dataset_path}.")

    
    
    prompts = []
    logger.info(f"First sample:\n{dataset[0]}")
    logger.info(f"Selecting {args.mode} mode.")
    hits = 0
    for i, ((question_id, item)) in enumerate(tqdm(dataset)): 
    
        question = item['question']
        options = item['options']
        gold_answer = item['answer'] if args.mode != "no-symbols" else item['options'][item['answer']]
        given_answer = id2answer_open[question_id]
        user_input = f"""Question: "{question.strip()}"\n\nOpen-ended Answer: "{given_answer}"\n\nPossible Options:\nA) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D'].strip()}"""
        
        
        #system_instruction = "You are a medical expert. Given an original question, a student’s open-ended answer, and five multiple-choice options (A–D and E for 'None of the above'), your task is to determine which option the student’s answer corresponds to. If the answer does not match any provided option, choose 'E' (None of the above). You MUST first provide a short explanation prefixed by '### Explanation:' and then return your final answer prefixed by '### Answer:'."
        prompt = f"""
You are a medical expert and evaluation judge. 
Your task is to map an open-ended student answer to the most appropriate multiple-choice option from the given question.

Instructions:
1. Carefully read the question.
2. Compare the student’s open-ended answer with the provided multiple-choice options.
3. Select the option that best matches in meaning, even if the wording differs.
4. If multiple options are similar, choose the one that is most precise and medically correct.
5. If the open-ended answer clearly corresponds to more than one option, list all possible mappings (e.g., "A or B").
6. If the answer is too vague or does not match any option, output "No clear match" with a short explanation.
7. Output only the option letter (e.g., "A"), or a list in case of multiple mappings, along with a short justification.

{user_input}

Your final answer must follow this Output Format:
- Mapped Option: [Letter or list of letters]
- Justification: [Brief reasoning]
"""

        messages = [
            #{"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt }
        ]
            
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        if i < 5:
            logger.info(f"Prompt:\n{text}")
            logger.info(f"*"* 10)

        prompts.append({
            "id": question_id, 
            "prompt": text.strip(), 
            "question": question,
            "options": options,
            "gold_answer": gold_answer,
            "given_answer": given_answer
        })
        #prompts.append((item['id'], text, messages))
    
    
    with open(f"{output_dir}/sample_prompts.txt", "w") as f:
        for prompt in prompts[:5]:
            f.write("--- PROMPT ---\n\n")
            f.write(prompt['prompt'])
            f.write("---------------\n\n")

    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    logger.info(f"Number of prompts: {len(prompts) * args.n_out_sequences}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0]) * args.n_out_sequences}")

    start_time_all = time.time()
    for id_batch, batch in enumerate(tqdm(batches)):
        
        ids = [el['id'] for el in batch]
        input_prompts = [el['prompt'] for el in batch]
        questions_batch = [el['question'] for el in batch]
        options_batch = [dict(el['options']) for el in batch]
        logger.info(f"Batch {id_batch} - options: {options}")
        golds = [el['gold_answer'] for el in batch]
        given_answers = [el['given_answer'] for el in batch]

        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

        for id_out, out in enumerate(outputs):
            
            #completions = [o.text.strip() for o in out.outputs]

            #for completion in completions:
            for o in out.outputs:
                completion = o.text.strip()
                logprobs = o.logprobs
                
                logger.info(f"Response:\n{completion}")
                parts = completion.split("</think>", 1)
                thinking = parts[0]
                completion = parts[1] if len(parts) > 1 else ""
                #thinking = completion.split("</think>")[0] if "</think>" in completion else completion
                #completion = completion.split("</think>")[1] if "</think>" in completion else ""
                final_answer = extract_answer(completion) if completion else None
                justification = completion.split("Justification:")[1].strip() if "Justification:" in completion else ""
                
                # if args.strategy == "direct" and final_answer is not None:
                #     confidence = None
                #     for logprob in logprobs:
                #         for id_token, token_logprob in logprob.items():
                #             if token_logprob.rank == 1 and "answer" not in token_logprob.decoded_token.lower() and final_answer in token_logprob.decoded_token:
                #                 logger.info(f"Logprobs: {token_logprob.decoded_token} - {round(np.exp(token_logprob.logprob) * 100, 2)}")
                #                 confidence = round(np.exp(token_logprob.logprob) * 100, 2)
                #                 break
        
                gold_answer = golds[id_out]
                question_id = ids[id_out]
                question = questions_batch[id_out]
                options = options_batch[id_out]
                given_answer = given_answers[id_out]

                if final_answer is None:
                    logger.warning(f"Unable to extract answer from question: {question_id}")

                    with open(f"{output_dir}/fails.jsonl", "a") as f:
                        fail_dict = {"question_id": question_id, "gold_answer": gold_answer}
                        json.dump(fail_dict, f)
                        f.write("\n")
                    continue
                
                elif final_answer == "No clear match":
                    

                    check_open_answer_prompt = f"""
You are an expert examiner. You will receive:
- The original multiple-choice question (MCQ) with options and the correct option.
- A student's open-ended answer, given without being shown the options and which does not clearly match any of the provided options.

Your task is to evaluate whether the student's answer is a **valid alternative answer** to the question, based on standard knowledge. 
Ignore the MCQ options — focus only on whether the answer could reasonably be accepted as correct in concept, even though it is not one of the listed options.
 
Return your evaluation in JSON format as follows:
```json 
{{
  "valid_alternative": "Yes" | "Partially" | "No",
  "justification": "A brief explanation why the student's answer is or is not a valid alternative answer."
}}
```

Now evaluate the following:
 
Original Question: {question}
Options: {options}
Correct Option: {gold_answer}
Student Answer: {given_answer}
"""

                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[{"role": "user", "content": check_open_answer_prompt}],
                        extra_body={
                        'extra_body': {
                            "google": {
                            "thinking_config": {
                                "thinking_budget": 1024,
                                "include_thoughts": True
                            }
                            }
                        }
                        }
                    )
                    print("-----------")
                    print(response)
                    print(response.choices[0].message)
                    print("-----------")
                    judge_completion = response.choices[0].message.content
                    judge_thinking = judge_completion.split("</thought>", 1)[0]
                    judge_answer = judge_completion.split("</thought>", 1)[1] if "</thought>" in judge_completion else ""
                    json_block = extract_json_block(judge_answer)
                    if json_block is None:
                        json_block = {"valid_alternative": "No", "justification": "No valid JSON block found in the response."}
                    valid_alternative = json_block.get("valid_alternative", "No")
                    judge_justification = json_block.get("justification", "")
                    is_valid_aternative = valid_alternative

                    resp_dict = response.model_dump()
                    with open(f"{output_dir}/judges_alternative_{args.subset}.jsonl", "a") as f:
                        json.dump(resp_dict, f)
                        f.write("\n")
                
                logger.info(f"Final answer: {final_answer}, Gold answer: {gold_answer}")
        
                if final_answer != "No clear match":
                    correct = str(final_answer) == str(gold_answer)
                else:
                    correct = is_valid_aternative.lower() == "yes"

                mcq_response = id2answer_mcq[question_id]
                consistent = str(final_answer) == str(mcq_response)

                if correct:
                    hits += 1
            
                with open(f"{output_dir}/judges_{args.subset}.jsonl", "a") as f:
                    compl_dict = {"question_id": question_id, "gold_answer": gold_answer, "mcq_response": mcq_response, "open_response": final_answer, "consistent": consistent, "completion": completion, "correct": correct, "justification": justification, "valid_alternative": is_valid_aternative if final_answer == "No clear match" else None, "justification_alternative": judge_justification if final_answer == "No clear match" else None}
                    if args.strategy == "direct":
                        compl_dict["confidence"] = confidence
                    json.dump(compl_dict, f)
                    f.write("\n")
    
    
    accuracy = hits / len(dataset) * 100
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Total Hits: {hits}")
    logger.info(f"Total Samples: {len(dataset)}")

    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Completions saved to: {output_dir}/completions.jsonl")


     