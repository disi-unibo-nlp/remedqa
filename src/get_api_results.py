import json
from google import genai
from google.genai import types
from openai import OpenAI
import time
from dotenv import load_dotenv
import os
load_dotenv()

import re

def parse_open_answer(response: str):
    """
    Extracts the open-ended answer from a response containing 'Final Answer: <answer>'.
    
    Handles cases like:
    - 'Final Answer: The answer is 42.'
    - 'Final Answer: 42'
    - 'Answer: The capital of France is Paris.'
    Returns the answer as a string, or None if not found.
    """
    match = re.search(r"(?:Final Answer:|Answer:)\s*(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def parse_mcq_answer(text: str) -> str:
    """
    Extracts the answer letter from strings like:
    'The final answer is \\boxed{C}' or 'Final Answer: (C)'
    Returns 'C'
    """
    # Try LaTeX boxed format: \boxed{C}
    match = re.search(r'\\boxed\{([A-D])\}', text)
    if match:
        return match.group(1)
    
    # Try parentheses format: (C)
    match = re.search(r'\(([A-D])\)', text)
    if match:
        return match.group(1)
    
    return None  # Nothing matched

def parse_final_answer(text: str, mode: str) -> str:
    if mode == "mcq":
        return parse_mcq_answer(text)
    elif mode == "open":
        return parse_open_answer(text)
    return None


def save_results_gemini(job_name, output_dir):

    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30) # Wait for 30 seconds before polling again
        batch_job = client.batches.get(name=job_name)

    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")


    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step

    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            print(f"Results are in file: {result_file_name}")

            print("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            # Process file_content (bytes) as needed
            print(file_content.decode('utf-8'))

            with open(f"{output_dir}/raw_completions.jsonl", "w", encoding="utf-8") as f:
                f.write(file_content.decode("utf-8"))

            with open(f"{output_dir}/raw_completions.jsonl") as f:
                completions = [json.loads(line) for line in f.readlines()]
            
            with open(f"{output_dir}/generations.jsonl", "w") as f:
                for item in completions:
                    key = item['key']
                    key_splits = key.split("-", 2)
                    dataset_item = key_splits[0]
                    mode_item = key_splits[1]
                    id_item = key_splits[2]
                    parts = item['response']['candidates'][0]['content']['parts'] if 'candidates' in item['response'] else []
                    thinking = ""
                    answer = ""
                    final_answer = ""
                    for part in parts:
                        if 'thought' in part:
                            thinking = part['text']
                        else:
                            answer = part['text']
                            final_answer = parse_final_answer(answer, mode_item)
                    
                    json.dump({"id_question": id_item, "dataset": dataset_item, "mode": mode_item, "final_answer": final_answer, "completion": answer, "thinking": thinking}, f)
                    f.write("\n")
            

        # If batch job was created with inline request
        # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
        elif batch_job.dest and batch_job.dest.inlined_responses:
            # Results are inline
            print("Results are inline:")
            for i, inline_response in enumerate(batch_job.dest.inlined_responses):
                print(f"Response {i+1}:")
                if inline_response.response:
                    # Accessing response, structure may vary.
                    try:
                        print(inline_response.response.text)
                    except AttributeError:
                        print(inline_response.response) # Fallback
                elif inline_response.error:
                    print(f"Error: {inline_response.error}")
        else:
            print("No results found (neither file nor inline).")
    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")


def save_results_openai(job_name, output_dir):

    

    #print(client.batches.retrieve(args.batch_id))
    response = client.batches.retrieve(job_name)
    is_safe = False
    if response.status =='completed':
        print("INFERENCE COMPLETED!")
        print(response)

        if response.error_file_id:
            print("ERROR FILE ID: ", response.error_file_id)
            file_response = client.files.content(response.error_file_id)
            for line in file_response.text.splitlines():
                print(line)
        elif response.output_file_id:
            is_safe = True
            print("OUTPUT FILE ID: ", response.output_file_id)
            file_response = client.files.content(response.output_file_id)
            out_file_id = response.output_file_id
        
            for line in file_response.text.splitlines():
                print(line)
            print("Saving results...")
            with open(f'{output_dir}/raw_completions.jsonl', 'w') as f:
                for line in file_response.text.splitlines():
                    json.dump(json.loads(line), f, ensure_ascii=False)
                    f.write('\n')
            print("Done!")

        if is_safe:
            print("Parsing results...")
            with open(f'{output_dir}/raw_completions.jsonl', 'r') as f:
                completions = [json.loads(line) for line in f.readlines()]
            
            with open(f'{output_dir}/generations.jsonl', 'w') as f:
                for item in completions:

                    key_splits = item['custom_id'].split("-", 2)
                    dataset_item = key_splits[0]
                    mode_item = key_splits[1]
                    id_item = key_splits[2]
                    usage_info = item['response']['body']['usage']['output_tokens_details']['reasoning_tokens']
                    #result = json.loads(line)
                    output = item['response']['body']['output']
                    final_answer = ""
                    thinking = ""
                    for out in output:
                        if out['type'] == "reasoning":
                            thinking = out['summary'][0]['text'] if out['summary'] else ""
                        elif out['type'] == "message":
                            completion = out['content'][0]['text'] if out['content'] else ""
                            final_answer = parse_final_answer(completion, mode_item)
                
                    json.dump({"id_question": id_item, "dataset": dataset_item, "mode": mode_item, "final_answer": final_answer, "completion": completion, "thinking": thinking, "thinking_length": usage_info}, f)
                    f.write("\n")
            print("Done!")

    else:
        print("BATCH STILL PROCESSING...")
        print(f"STATUS: {response.status}")

if __name__ == "__main__":
    
    
    # Use the name of the job you want to check
    # e.g., inline_batch_job.name from the previous step
    job_name = "batch_68d9d135db74819096f0dd2286121b87" #"batches/jdsea2vgjodl3ftrdvxgpp3f478wwteg01ld" #"batches/taxgnuxeblk3sq3hkgxc6pmskwbaoqtx74kt"  # (e.g. 'batches/your-batch-id')
    output_dir = "out/completions/openai_api/gpt-5-mini/mmlu/2025-09-29_00-22-11"

    # job_name = "batches/r9nzb8h32kwwg61417wevmw260ezpi1ylen2" #"batches/jdsea2vgjodl3ftrdvxgpp3f478wwteg01ld" #"batches/taxgnuxeblk3sq3hkgxc6pmskwbaoqtx74kt"  # (e.g. 'batches/your-batch-id')
    # output_dir = "out/completions/gemini_api/gemini-2.5-flash/medmcqa/2025-09-28_20-21-06"

    if "gemini" in output_dir:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=GEMINI_API_KEY)

        batch_job = client.batches.get(name=job_name)
        print("Processing results...")
        save_results_gemini(job_name, output_dir)
        print("Done!")
    elif "openai" in output_dir:
        OPENAI_KEY = os.getenv("OPENAI_KEY")
        client = OpenAI(api_key=OPENAI_KEY)

        save_results_openai(job_name, output_dir)
   