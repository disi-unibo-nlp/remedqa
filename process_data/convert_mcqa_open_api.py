import json
import os
import datetime
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SYSTEM_INSTRUCTION = f"""
You are given a <DATASET> multiple-choice question. Rewrite it into an open-ended question using the following rules:

1. Keep the clinical vignette exactly as it is. Do not shorten, paraphrase, or change it in any way.  

2. Only replace the final question sentence with an equivalent open-ended version that asks directly for the answer.  
   - The rewritten question must allow a clear answer that can be directly mapped to one of the original options without introducing new information.  
   - Phrases like "most likely" or "most appropriate" are allowed if they still point to a single correct answer.  
   - Be sure to remove the original final question sentence
   - Enclose the rewritten final question in << >>.

3. Before rewriting, check if the question CANNOT be adapted into open-ended form. If ANY of the following is true, output only "Not possible":  
   A. The question uses NOT, EXCEPT, or asks for exclusions.  
   B. If the correct answer can only be determined by subtle differences among the listed options
   C. The question requires comparing multiple options, ranking, or ordering (e.g., "first-line vs second-line," "most/least effective among multiple options").  
   D. The question requires matching items, relative likelihoods, or prioritization across multiple options.  
   E. The question would become vague or ambiguous if rewritten.  
   F. The question would allow more than one correct answer if the options were removed.

4. Return ONLY the following, in order, using these exact prefixes:  
   - Rationale: <short 1–2 sentence explanation of whether the question can be rewritten>  
   - Output: <either the original clinical vignette plus the rewritten final sentence in << >>, or "Not possible">  
   Do NOT add any text, comments, or formatting beyond these two lines.

---

Examples

Original:
A junior orthopaedic surgery resident is ... out of the operative report. Which of the following is the correct next action for the resident to take?
(A) Disclose the error to the patient and put it in the operative report
(B) Tell the attending that he cannot fail to disclose this mistake
(C) Report the physician to the ethics committee
(D) Refuse to dictate the operative report

Rationale: The question asks for a single best answer and does not require ranking or exclusions.
Output: A junior orthopaedic surgery resident is ... out of the operative report. <<What is the correct next action for the resident to take?>>

---

Original:
A 24-year-old G2P1 woman ... Which of the following is most true about the confirmatory test?
(A) It is a Southwestern blot, identifying the presence of DNA-binding proteins
(B) It is a Northern blot, identifying the presence of RNA
(C) It is a Northern blot, identifying the presence of DNA
(D) It is an HIV-1/HIV2 antibody differentiation immunoassay

Rationale: The question requires comparing multiple options. Also, it can only be determined by the subtle differences among the listed options. So it cannot be rewritten clearly.
Output: Not possible"""

SYSTEM_INSTRUCTION_MEDMCQA = """
You are given multiple-choice medical exam questions from the MedMCQA dataset. 
Your task is to convert them into open-ended questions ONLY if this can be done unambiguously 
(i.e., the question can be answered without relying on the options, and the open-ended version has a single clear correct answer).

### Instructions:
1. **Check convertibility:**
   - If the question requires the options to determine the correct answer 
     (e.g., "Which of the following...", "All of the following except...", or negation-based questions), 
     mark it as **Not possible** and provide a short rationale (e.g., "needs options", "negation-based", "multiple possible answers", or "ambiguous without options").
   - If the question can be answered correctly without the options (e.g., fact-based or definition-type questions), 
     CONVERT it into an open-ended form.

2. **Conversion rules:**
   - Remove all multiple-choice options.
   - Rewrite the question so it stands alone as an open-ended medical question.
   - The rewrite can involve changing the entire question or only part of it: in both cases, enclose the rewritten text clearly within << >>.
   - Ensure the wording makes sense without options.


3. **Output format:**
   - If convertible:
     Rationale: <short 1–2 sentence explanation of why it can be converted>
     Output: <open-ended question with the changed part enclosed in << >>, without options>
   - If not convertible:
     Rationale: <short 1–2 sentence explanation of why it cannot be converted>
     Output: Not possible

### Examples:

Original:
Which of the following is not true for myelinated nerve fibers:  
(A) Impulse through myelinated fibers is slower than non-myelinated fibers  
(B) Membrane currents are generated at nodes of Ranvier  
(C) Saltatory conduction of impulses is seen  
(D) Local anesthesia is effective only when the nerve is not covered by myelin sheath  

Rationale: The question uses negation ("not true") and requires options to determine the correct answer.
Output: Not possible

---

Original: 
Axonal transport is:  
(A) Antegrade  
(B) Retrograde  
(C) Antegrade and retrograde  
(D) None  

Rationale: The question can be answered without options as it asks for types of axonal transport.
Output: <<What types of axonal transport are observed in neurons?>>

---

Original: 
A blue new born presents with cyanosis. The X-ray chest reveal oligaemic lung field and normal sized heart. Most likely diagnosis is:
(A) Ebstein's anomaly
(B) Pulmonary atresia
(C) Transposition of great arteries
(D) Tetralogy of fallot

Rationale: The question can be answered without options as it asks for the most likely diagnosis based on clinical presentation.
Output: A blue new born presents with cyanosis. The X-ray chest reveal oligaemic lung field and normal sized heart. <<What is the most likely diagnosis?>>"""

def initialize_openai_client() -> OpenAI:
    """Initialize and return OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

def make_completion_openai(client: OpenAI, system_instruction: str, prompt: str, model_name: str) -> Tuple[str, str, Dict[str, int]]:
    """Make a completion request to OpenAI API."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    completion = response.choices[0].message.content
    print(completion)
    rationale = completion.split("Output:")[0].replace("Rationale:", "").strip()
    completion = completion.split("Output:")[1].strip()
    
    usage_info = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    return completion, rationale, usage_info

def load_dataset(dataset_path: str, dataset_name: str) -> list:
    """Load and prepare dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    dataset = data[dataset_name.lower()]
    return [(idx, item) for idx, item in dataset.items()]

def format_question(item: Dict, dataset_name: str) -> Tuple[str, str]:
    """Format question with options and create prompt."""
    question = item["question"].strip()
    options = item["options"]
    formatted_question = f"{question}\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n(D) {options['D'].strip()}\n\n"
    prompt = f"Rewrite the following {dataset_name} question into an open-ended version:\n\n{formatted_question}"
    return formatted_question, prompt

def create_batch_request(model_name: str, system_instruction: str, prompt: str, custom_id: str, temperature: float = 0.0) -> Dict:
    """Create a batch request object."""
    base_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
        }
    }
    
    if "gpt-4" in model_name:
        base_request["body"]["temperature"] = temperature
    elif "o4-mini" in model_name:
        base_request["body"]["reasoning_effort"] = "low"
    
    return base_request

def run_batch_mode(client: OpenAI, dataset: list, args: argparse.Namespace) -> None:
    """Run batch processing mode."""
    output_dir = Path(args.output_dir) / "openai_batch" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_prompts = 0
    
    batch_file_path = output_dir / "input_batch.jsonl"

    if args.dataset_name.lower() == "medqa":
        system_instruction = SYSTEM_INSTRUCTION.replace("<DATASET>", "MedQA")
    elif args.dataset_name.lower() == "medmcqa":
        system_instruction = SYSTEM_INSTRUCTION_MEDMCQA
    elif args.dataset_name.lower() == "mmlu":
        system_instruction = SYSTEM_INSTRUCTION_MEDMCQA.replace("MedMCQA", "MMLU")
    
    with open(batch_file_path, 'w') as f:
        for i, (idx, item) in enumerate(tqdm(dataset, desc="Creating batch requests")):
            question, prompt = format_question(item, args.dataset_name)
            custom_id = f"request-{idx}"
            
            batch_request = create_batch_request(
                args.model_name, 
                system_instruction, 
                prompt, 
                custom_id, 
                args.temperature
            )
            
            json.dump(batch_request, f, ensure_ascii=False)
            f.write("\n")
            total_prompts += 1
    
    print(f"TOTAL PROMPTS: {total_prompts}")
    
    # Upload batch file
    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    
    # Create batch job
    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Running batch inference for {args.dataset_name} parsing."
        }
    )
    
    print(f"Batch created: {batch_obj}")
    print(f"BATCH ID: {batch_obj.id}")
    
    # Save batch ID
    with open(output_dir / "batch_id.txt", 'w') as f:
        f.write(batch_obj.id)

def run_direct_mode(client: OpenAI, dataset: list, args: argparse.Namespace) -> None:
    """Run direct processing mode."""
    if args.dataset_name.lower() == "medqa":
        system_instruction = SYSTEM_INSTRUCTION.replace("<DATASET>", "MedQA")
    elif args.dataset_name.lower() == "medmcqa":
        system_instruction = SYSTEM_INSTRUCTION_MEDMCQA
    elif args.dataset_name.lower() == "mmlu":
        system_instruction = SYSTEM_INSTRUCTION_MEDMCQA.replace("MedMCQA", "MMLU")

    results = {}
    
    for idx, item in tqdm(dataset, desc="Processing questions"):
        question, prompt = format_question(item, args.dataset_name)
        
        # try:
        rewritten, rationale, usage_info = make_completion_openai(
            client, system_instruction, prompt, args.model_name
        )
        
        print(f"Rewritten: {rewritten}")
        print(f"Rationale: {rationale}")
        change_idx = rewritten.find("<<")
        unchanged_part = question[:change_idx].strip() if change_idx != -1 else question
        unchanged_part_rewritten = rewritten[:change_idx].strip() if change_idx != -1 else rewritten
        
        results[idx] = {
            "original": question,
            "rewritten": rewritten.replace("<<", "").replace(">>", ""),
            "rationale": rationale,
            "usage": usage_info,
            "safe": unchanged_part.strip() == unchanged_part_rewritten.strip(),
            "changed_part": question[change_idx:].strip() if "<<" in rewritten else "Not possible",
            "new_part": rewritten[change_idx:].replace("<<", "").replace(">>", "").strip() if "<<" in rewritten else "Not possible"
        }
        
        if args.verbose:
            print("-----")
            print(f"Processed question {idx}")  
            print("-----")
            print(f"Original: {question}")
            print("-----")
            print(f"Rewritten: {rewritten}")
            print("-----")
            print(f"Safe: {results[idx]['safe']}")
            print("\n\n")
                
        # except Exception as e:
        #     print(f"Error processing question {idx}: {e}")
        #     continue
    
    # Save results
    output_file = f"benchmark_rewritten_{args.model_name}_{args.dataset_name.lower()}_prova.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function to run the question rewriter."""
    parser = argparse.ArgumentParser(description="Rewrite dataset questions using OpenAI API")
    
    # Required arguments
    parser.add_argument("--dataset-name", type=str, default="MedMCQA", 
                       help="Dataset name (default: MedQA)")
    parser.add_argument("--model-name", type=str, default="gpt-4.1", 
                       help="OpenAI model name (default: gpt-4.1)")
    
    # API mode
    parser.add_argument("--api-mode", type=str, choices=["batch", "direct"], default="batch",
                       help="API mode: batch or direct (default: direct)")
    
    # Optional arguments
    parser.add_argument("--dataset-path", type=str, default="data/benchmark.json",
                       help="Path to dataset file (default: data/benchmark.json)")
    parser.add_argument("--output-dir", type=str, default="out",
                       help="Output directory (default: out)")
    parser.add_argument("--system-instruction", type=str, default=None,
                       help="System instruction for the model")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for generation (default: 0.0)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of questions to process (for testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output during processing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    
    # Initialize client
    try:
        client = initialize_openai_client()
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}")
        return 1
    
    # Load dataset
    try:
        dataset = load_dataset(args.dataset_path, args.dataset_name)
        # Limit dataset if specified
        if args.limit:
            dataset = dataset[:args.limit]
        print(f"Loaded {len(dataset)} questions from {args.dataset_name}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Run processing
    # try:
    if args.api_mode == "batch":
        run_batch_mode(client, dataset, args)
    else:
        run_direct_mode(client, dataset, args)
    # except Exception as e:
    #     print(f"Error during processing: {e}")
    #     return 1
    
    return 0

if __name__ == "__main__":
    exit(main())