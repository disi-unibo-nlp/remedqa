import json
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
from together import Together
import argparse


def get_thinking_budget(effort):
    thinking_budget = {
        "low": 1024, 
        "medium": 8192, 
        "high": 24576
    }
    return thinking_budget[effort]

def create_batch_gemini(modes, subset, output_dir, reasoning_effort="low", model_name_path="gemini-2.5-flash"):
    # Create a sample JSONL file
    if option_only_enabled:
        data_path = f"data/processed/{subset}_options_only.json"
    else:
        data_path = f"data/processed/{subset}_all_think.json"
    benchmark = json.load(open(f"{data_path}"))
    
    thinking_budget = get_thinking_budget(reasoning_effort)
    with open(f"{output_dir}/my-batch-requests.jsonl", "w") as f:
        for mode in modes:
            data = benchmark[mode]
            for count, (idx, item) in enumerate(data.items()):
                #if count < 20:
                request = {"key": f"{subset}-{mode}-{idx}", "request": {"contents": [{"parts": [{"text": item['prompt']}]}], "generation_config": {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": thinking_budget} }}}
                f.write(json.dumps(request) + "\n")

    # Upload the file to the File API
    uploaded_file = client.files.upload(
        file=f'{output_dir}/my-batch-requests.jsonl',
        config=types.UploadFileConfig(display_name=f'my-batch-requests-{now_dir}', mime_type='jsonl')
    )

    print(f"Uploaded file: {uploaded_file.name}")


    # Assumes `uploaded_file` is the file object from the previous step
    file_batch_job = client.batches.create(
        model=model_name_path,
        src=uploaded_file.name,
        config={
            'display_name': f"file-upload-job-{now_dir}",
        },
    )

    print(f"Created batch job: {file_batch_job.name}")
    with open(f"{output_dir}/jod_id.txt", "w") as f:
        f.write(f"{file_batch_job.name}")


def create_batch_openai(modes, subset, output_dir, reasoning_effort="low", model_name_path="gpt-5-mini"):
    """Create a batch request object."""
    if option_only_enabled:
            data_path = f"data/processed/{subset}_options_only.json"
    else:
        data_path = f"data/processed/{subset}_all_think.json"
    benchmark = json.load(open(f"{data_path}"))
    batch_input_file = f"{output_dir}/my-batch-requests.jsonl"

    with open(batch_input_file, "w") as f:
        for mode in modes:
            data = benchmark[mode]
            for count, (idx, item) in enumerate(data.items()):
                #if count < 20:
                
                request = {
                    "custom_id": f"{subset}-{mode}-{idx}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model_name_path,
                        "input": item['prompt'],
                    }
                }
    
                if "gpt-4" in model_name_path:
                    request["body"]["temperature"] = 0
                elif "gpt-5" in model_name_path:
                    request["body"]["reasoning"] =  {
                        "effort": reasoning_effort,
                        "summary": "detailed"
                    }
                f.write(json.dumps(request) + "\n")
    
    # Upload batch file
    batch_input_file = client.files.create(
        file=open(batch_input_file, "rb"),
        purpose="batch"
    )
    
    # Create batch job
    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": f"Running batch inference for {subset} evaluation."
        }
    )
    
    print(f"Batch created: {batch_obj}")
    print(f"BATCH ID: {batch_obj.id}")
    
    # Save batch ID
    with open(f"{output_dir}/job_id.txt", 'w') as f:
        f.write(batch_obj.id)



def create_batch_together(modes, subset, output_dir, reasoning_effort="medium", model_name_path=None):
    # Create a sample JSONL file

    if "gpt-oss" in model_name_path.lower():
        if option_only_enabled:
            data_path = f"data/processed/{subset}_options_only.json"
        else:
            data_path = f"data/processed/{subset}_all_think.json"
    else:
        data_path = f"data/processed/{subset}_all.json"
    benchmark = json.load(open(f"{data_path}"))
    
    # Create a sample JSONL file
    model_name_request = model_name_path
    model_name = model_name_request.split("/")[-1]
    with open(f"{output_dir}/batch_togther_{model_name}.jsonl", "w") as f:
        for mode in modes:
            data = benchmark[mode]
            for count, (idx, item) in enumerate(data.items()):
                prompt = item['prompt']
                #if count < 10:
                if "gpt-oss" in model_name_path:
                    request = {"custom_id": f"{subset}-{mode}-{idx}", "body": {"model": model_name_request, "messages": [{"role": "user", "content": prompt}], "reasoning_effort": reasoning_effort}}
                else:
                    request = {"custom_id": f"{subset}-{mode}-{idx}", "body": {"model": model_name_request, "messages": [{"role": "system", "content": "You are a medical expert."}, {"role": "user", "content": prompt}], "temperature": 0}}

                f.write(json.dumps(request) + "\n")

    file_resp = client.files.upload(file=f"{output_dir}/batch_togther_{model_name}.jsonl", purpose="batch-api")

    file_id = file_resp.id

    batch = client.batches.create_batch(file_id, endpoint="/v1/chat/completions")

    batch_stat = client.batches.get_batch(batch.id)

    print(batch.id)
    print(f"Created batch job: {batch.id}")
    with open(f"{output_dir}/jod_id.txt", "w") as f:
        f.write(f"{batch.id}")

    print(batch_stat.status)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and modes.")
    
    parser.add_argument(
        "--subset",
        type=str,
        default="mmlu",
        choices=["mmlu", "medmcqa", "medqa"],
        help="Dataset subset to use (mmlu, medqa, or medmcqa)."
    )

    parser.add_argument(
        "--option-only-mode",
        action="store_true",
        help="If set, restricts evaluation to option-only modes (overrides --modes)."
    )

    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=[
            "open", 
            "mcq", 
            "incorrect", 
            "none_of_the_provided", 
            "roman_numeral", 
            "fixed_pos", 
            "no_symbols",
            "options_only_mcq", 
            "options_only_incorrect", 
            "options_only_roman_numeral", 
            "options_only_none_of_the_provided", 
            "options_only_fixed_pos", 
            "options_only_no_symbols"
        ],
        default=None,
        help=(
            "List of modes to evaluate. "
            "If not provided, defaults are determined by --option_only_enabled. "
            "Examples: --modes open mcq incorrect"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Name or path of the model to evaluate."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions",
        help="Output directory to save results."
    )

    args = parser.parse_args()


    now = datetime.now()
    # Format the date and time as a string
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    option_only_enabled = args.option_only_mode
    subset = args.subset
    model_name = args.model_name

    # Determine modes
    if args.modes is not None:
        modes = args.modes
    elif option_only_enabled:
        modes = ['options_only_mcq', 'options_only_incorrect', 'options_only_roman_numeral', 
                    'options_only_none_of_the_provided', 'options_only_fixed_pos', 'options_only_no_symbols']
            
    else:
        modes = ["open", "mcq", "incorrect", "none_of_the_provided", "roman_numeral", "fixed_pos", "no_symbols"] ##, "open"] 

    if "gemini" in model_name:
        api_dir = "gemini" 
    elif "gpt-5" in model_name:
        api_dir = "openai"
    else:
        api_dir = "together"

    output_dir = f"{args.output_dir}/{api_dir}_api/{model_name}/{subset}/{now_dir}"
    os.makedirs(output_dir, exist_ok=True)
    load_dotenv()


    if "gemini" in model_name:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        client = genai.Client(api_key=GEMINI_API_KEY)
        create_batch_gemini(
            modes=modes,
            subset=subset,
            output_dir=output_dir,
            reasoning_effort="low",
            model_name_path=model_name
        )
    elif "gpt-5-mini" in model_name:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=OPENAI_API_KEY)
        create_batch_openai(
            modes=modes,
            subset=subset,
            output_dir=output_dir,
            reasoning_effort="low",
            model_name_path=model_name
        )

    else:
        TOGETHER_API_KEY=os.getenv("TOGETHER_API_KEY")
        if not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        client = Together(api_key=TOGETHER_API_KEY) 
        create_batch_together(
            modes=modes,
            subset=subset,
            output_dir=output_dir,
            reasoning_effort="medium",
            model_name_path=model_name
        )