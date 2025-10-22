from openai import OpenAI

import os
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
# parse input args
   

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script arguments for batch result retrieval")

    parser.add_argument("--batch_id", type=str, default="", help="batch id to retrieve from OpenAI API")
    parser.add_argument("--out_dir", type=str, default="out/openai_batch/20250909_100941", help="directory where to store results.")
    parser.add_argument("--dataset-name", type=str, default="medmcqa", choices=["medqa", "mmlu", "medmcqa"], help="Dataset name (default: medqa)")
    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    with open('data/benchmark.json', 'r') as f:
        data = json.load(f)

    args = parse_args()
    
    dataset = data[args.dataset_name]
    dataset = [(idx, item) for idx, item in dataset.items()]

    id2question = {idx: item['question'] for idx, item in dataset}

    

    #print(client.batches.retrieve(args.batch_id))
    response = client.batches.retrieve(args.batch_id)
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
            for line in file_response.text.splitlines():
                with open(f'{args.out_dir}/raw_completions.jsonl', 'a') as f:
                    json.dump(json.loads(line), f, ensure_ascii=False)
                    f.write('\n')
            print("Done!")
    else:
        print("BATCH STILL PROCESSING...")
        print(f"STATUS: {response.status}")
    
    if is_safe:
        print("Parsing results...")
        with open(f'{args.out_dir}/raw_completions.jsonl', 'r') as f:
            completions = [json.loads(line) for line in f.readlines()]
        results = {}
        for item in completions:
            #result = json.loads(line)
            completion = item['response']['body']['choices'][0]['message']['content']
            usage_info = item['response']['body']['usage']
            id_problem = str(item['custom_id'].split('-', 1)[1].strip())
            question = id2question[id_problem].strip()
            rationale = completion.split("Output:")[0].replace("Rationale:", "").strip()
            rewritten = completion.split("Output:")[1].strip()

            change_idx = rewritten.find("<<")
            unchanged_part = question[:change_idx].strip() if change_idx != -1 else question
            unchanged_part_rewritten = rewritten[:change_idx].strip() if change_idx != -1 else rewritten
            
            results[id_problem] = {
                "original": question,
                "rewritten": rewritten.replace("<<", "").replace(">>", "").strip(),
                "rationale": rationale,
                "safe": unchanged_part.strip() == unchanged_part_rewritten.strip(),
                "changed_part": question[change_idx:].strip() if "<<" in rewritten else "Not possible",
                "new_part": rewritten[change_idx:].replace("<<", "").replace(">>", "").strip() if "<<" in rewritten else "Not possible"
            }
            
                
        with open(f"{args.out_dir}/annotations.json", 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("Done!")
   