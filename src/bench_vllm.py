# import json
# from openai_harmony import (
#     HarmonyEncodingName,
#     load_harmony_encoding,
#     Conversation,
#     Message,
#     Role,
#     SystemContent,
#     DeveloperContent,
# )
 
# from vllm import LLM, SamplingParams
 
# # --- 1) Render the prefill with Harmony ---
# encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 
# convo = Conversation.from_messages(
#     [
#         Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
#         Message.from_role_and_content(
#             Role.DEVELOPER,
#             DeveloperContent.new().with_instructions("Always respond in riddles"),
#         ),
#         Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
#     ]
# )
 
# prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
 
# # Harmony stop tokens (pass to sampler so they won't be included in output)
# stop_token_ids = encoding.stop_tokens_for_assistant_actions()
 
# # --- 2) Run vLLM with prefill ---
# llm = LLM(
#     model="openai/gpt-oss-20b",
#     trust_remote_code=True,
# )
 
# sampling = SamplingParams(
#     max_tokens=128,
#     temperature=1,
#     stop_token_ids=stop_token_ids,
# )
 
# outputs = llm.generate(
#     prompt_token_ids=[prefill_ids],   # batch of size 1
#     sampling_params=sampling,
# )
 
# # vLLM gives you both text and token IDs
# gen = outputs[0].outputs[0]
# text = gen.text
# output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)
 
# # --- 3) Parse the completion token IDs back into structured Harmony messages ---
# entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
 
# # 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
# for message in entries:
#     print(f"{json.dumps(message.to_dict())}")


# from dotenv import load_dotenv
# import os
# from together import Together
# load_dotenv()  # take environment variables from .env.
# client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")

# response = client.chat.completions.create(
#     model="openai/gpt-oss-20b",
#     messages=[
#       {
#         "role": "user",
#         "content": "What are some fun things to do in New York?"
#       }
#     ]
# )
# print(response.choices[0].message.content)

"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import os
import re
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple, Optional
from dotenv import load_dotenv  
load_dotenv()  # take environment variables from .env.
from huggingface_hub import login
login(token=os.environ.get("HUGGINGFACE_TOKEN"))
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils import FlexibleArgumentParser
from tqdm import tqdm

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    id_prompts: list[str]
    mode_prompts: list[str]
    data_info: Optional[list[dict]] = None
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None


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
    match = re.search(r"(?:Final Answer:|Answer:)\s*\(?([A-E])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_mcq_answer_roman(response: str, convert_roman=True):
    """
    Extracts the MCQ letter from a response containing 'Final Answer: (X)'.
    
    Handles cases like:
    - 'Final Answer: (I)'
    - 'Final Answer: I'
    - 'Answer: (II)'
    - 'Final Answer: III'
    Returns the option letter as a string (e.g., 'A'), or None if not found.
    """
    if convert_roman:
        #roman_to_letter = { 'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D' }
        match = re.search(r"(?:Final Answer:|Answer:)\s*\(?([IVX]+)\)?", response, re.IGNORECASE)
        if match:
            roman_numeral = match.group(1).upper()
            return roman_numeral
    else:
        match = re.search(r"(?:Final Answer:|Answer:)\s*\(?([A-D])\)?", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None

   

def extract_open_answer(response: str):
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

def extract_incorrect_answer(response: str):
    """
    Extracts one or more MCQ letters from a response after 'Final Answer:' or 'Answer:'.
    Handles cases like:
      - Final Answer: [A, C]
      - Final Answer: (A, B, C)
      - Final Answer: *A, B, C*
      - Final Answer: A, B, C
    Returns a list of option letters (['A','C']), or None if not found.
    """
    # regex: look after Final Answer: or Answer:
    # accept optional brackets (), [], or * * around the letters
    pattern = r'(?:Final Answer:|Answer:)\s*[\[\(\*\s]*([A-D](?:\s*,\s*[A-D])*)[\]\)\*\s]*'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        answers = [ans.strip().upper() for ans in match.group(1).split(",")]
        return answers
    return None

def extract_final_answer(response: str, mode: str):
    if mode in ['mcq', 'options_only', 'none_of_the_provided', 'fixed_pos', "idk_answer"]:
        return extract_mcq_answer(response)
    elif mode == "roman_numeral":
        return extract_mcq_answer_roman(response)
    elif mode == "incorrect":
        return extract_incorrect_answer(response)
    else:
        return extract_open_answer(response)

def run_qwen3_8b(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer

def run_llama3(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )
    
    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer


def run_jsl_medllama(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            f'''###Question: {question}\n###Answer:'''
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer

def run_phi(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer

def run_mediphi(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "microsoft/MediPhi-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer

def run_med42(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "m42-health/Llama3-Med42-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=True)
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer

def run_gemma3(questions: list[str], modality=None) -> ModelRequestData:
    model_name = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4000,
        max_num_seqs=1,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in model_name.lower() else "auto",
        quantization="awq" if "awq" in model_name.lower() else None,
        enforce_eager=False,
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], 
                tokenize=False, add_generation_prompt=True
            )
        )

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts, 
        data_info=data_info
    ), tokenizer

def run_medgemma(questions: list[str], modality=None) -> ModelRequestData:
    
    model_name = "google/medgemma-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=2048,
        max_num_seqs=1,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )

    prompts, id_prompts, mode_prompts, data_info = [], [], [], []

    for idx, question, mode, info in questions:
        prompts.append((
            "<bos><start_of_turn>user\n"
            f"<start_of_image>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ))

        id_prompts.append(idx)
        mode_prompts.append(mode)
        data_info.append(info)

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        id_prompts=id_prompts,
        mode_prompts=mode_prompts,
        data_info=data_info
    ), tokenizer


model_example_map = {
    "medgemma": run_medgemma,
    "gemma3": run_gemma3,
    "mediphi": run_mediphi,
    "Llama3-Med42-8B": run_med42,
    "JSL-MedLlama-3-8B-v2.0": run_jsl_medllama,
    "Qwen3-8B": run_qwen3_8b,
    "llama3": run_llama3,
    "Phi-3.5-mini": run_phi
}


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="llama3",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image"],
        help="Modality of the input.",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2000,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )

    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )

    parser.add_argument(
        "--use-different-prompt-per-request",
        action="store_true",
        help="If True, then use different prompt (with the same multi-modal "
        "data) for each request.",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/benchmark.json",
    )

    parser.add_argument(
        "--subset",
        type=str,
        choices = ["medqa", "medmcqa", "mmlu"],
        default="medqa",
    )

    parser.add_argument('--modes', nargs='+',
        choices=['mcq', 'open', 'incorrect', 'options_only', 'roman_numeral', 
                'yes_no_maybe', 'none_of_the_provided', 'fixed_pos', 'no_symbols', 'idk_answer'],
        help='Specific modes to process (default: all modes)'
    )

    parser.add_argument('--limit', type=int, default=None,
        help='Limit the number of questions to process (default: all)'
    )
    

    return parser.parse_args()


def main(args):
    import json
    with open(args.dataset_path) as f:
        dataset = json.load(f)#[args.subset]
    with open("data/benchmark_open.json") as f:
        dataset_open = json.load(f)[args.subset]

    modes = args.modes if args.modes else ["mcq", "open"]
    dataset_per_mode = {mode: dataset[mode] for mode in modes}
        
    # dataset = dataset["medqa"]
    non_possible_ids = [idx for (idx, item) in dataset_open.items() if item['open_question'] == 'Not possible']
    # dataset_open = [(idx, item) for idx, item in dataset_open.items() if idx not in non_possible_ids]
    # dataset = [(idx, item) for idx, item in dataset.items() if idx not in non_possible_ids]
    # assert len(dataset_open) == len(dataset)

    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    #modality = "image"
    #mm_input = get_multi_modal_input(args)
    questions = []
    for mode, dataset in dataset_per_mode.items():
        print(f"Processing mode: {mode} with {len(dataset)} questions")
        dataset = [(idx, item) for idx, item in dataset.items() if idx not in non_possible_ids]
        
        if args.limit:
            dataset = dataset[:args.limit]

        for idx, item in dataset:
            question = item['prompt']
            questions.append((idx, question, mode, item))

#     for item, item_open in zip(dataset, dataset_open):

#         id_question = item[0]
#         TEMPLATE = """The following are multiple choice questions about medical knowledge. 
# Solve them in a step-by-step fashion, starting by summarizing the available information. 
# Output a single option from the four options as the final answer. 
# Question: "<QUESTION>"

# Response (think step by step and then end with "Final Answer:" followed by *only* the letter corresponding to the correct answer enclosed in parentheses)"""

# #         TEMPLATE = """You have a maximum of 2000 tokens for your thinking process. 
# # Be concise and focused in your reasoning. When you approach this limit, conclude your thinking and provide your answer.

# # Question: "<QUESTION>"

# # Please show your choice in the answer field with only the choice letter, e.g., "Final Answer: (X)". Rember that your thinking must be concise."""        

#         TEMPLATE_OPEN = """The following are open-ended questions about medical knowledge.
# Solve them in a step-by-step fashion, starting by summarizing the available information.
# Output a single, concise final answer (not a letter).
# Question: "<QUESTION>"

# Response (think step by step and then end with "Final Answer:" followed by *only* the concise answer)"""
        
# #         TEMPLATE_OPEN = """You have a maximum of 2000 tokens for your thinking process. 
# # Be concise and focused in your reasoning. When you approach this limit, conclude your thinking and provide your answer.

# # Question: "<QUESTION>"

# # Please show your response in the answer field with only the concise final answer, e.g., "Final Answer: <your concise answer>". Rember that your thinking must be concise."""        

        
#         #gold_answer = item['answer'] if args.mode != "no-symbols" else item['options'][item['answer']]
#         question_open = f"{item_open[1]['open_question'].strip()}\n\n"
#         question_open = TEMPLATE_OPEN.replace("<QUESTION>", question_open)

#         question = item[1]['question']
#         options = item[1]['options']
#         question = f"{question.strip()}\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n(D) {options['D'].strip()}\n\n"
        
#         question = TEMPLATE.replace("<QUESTION>", question.strip()) 
#         questions.append((id_question, question))

#         questions.append((id_question, question_open))
    

    if "medgemma" in args.model_type.lower():
        req_data, tokenizer = model_example_map[model](questions)
        # Disable other modalities to save memory
        default_limits = {"image": 1, "video": 0, "audio": 0}
        req_data.engine_args.limit_mm_per_prompt = default_limits 

        engine_args = asdict(req_data.engine_args) | {
            "seed": args.seed,
            "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
        }
    else: 
        req_data, tokenizer = model_example_map[model](questions)
        engine_args = asdict(req_data.engine_args) | {
            "seed": args.seed
        }

    llm = LLM(**engine_args)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = req_data.prompts
    id_prompts = req_data.id_prompts
    mode_prompts = req_data.mode_prompts
    data_info = req_data.data_info

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.

    if "qwen3" in args.model_type.lower():
        sampling_params = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, max_tokens=32000, stop_token_ids=req_data.stop_token_ids
        )
    elif "JSL-MedLlama-3-8B-v2.0" in args.model_type:
        sampling_params = SamplingParams(
            temperature=0.7, top_k=50, top_p=0.95, max_tokens=2000, stop_token_ids=req_data.stop_token_ids
        )
    else:
        sampling_params = SamplingParams(
            temperature=0, max_tokens=args.max_new_tokens, stop_token_ids=req_data.stop_token_ids
        )

    # Batch inference
    
    if "medgemma" in args.model_type.lower():
        inputs = [

            {
                "meta_data": {
                    "id_prompt": id_prompts[i],
                    "mode_prompt": mode_prompts[i],
                    "answer": data_info[i]['answer']
                },
                "request": {   
                    "prompt": prompts[i],
                    "multi_modal_data": {"image": []},
                }
            }
            for i in range(len(prompts))
        ]
    else:
        inputs = [

            {
                "meta_data": {
                    "id_prompt": id_prompts[i],
                    "mode_prompt": mode_prompts[i],
                    "answer": data_info[i]['answer']
                },
                "request": prompts[i]
            }
            for i in range(len(prompts))
        ]


    batched_inputs = [inputs[i : i + args.batch_size] for i in range(0, len(inputs), args.batch_size)]

    # Add LoRA request if applicable
    lora_request = None

    print("Sample prompts:")
    os.makedirs(f"out/prompts/{args.model_type}", exist_ok=True)
    
    for prompt in inputs[:10]:
        mode_prompt = prompt['meta_data']['mode_prompt']
        with open(f"out/prompts/{args.model_type}/prompt_{args.subset}_{mode_prompt}.txt", "w") as f:
            f.write("-" * 50)
            f.write("\n")
            f.write(f'MODE_PROMPT: {prompt["meta_data"]["mode_prompt"]}\n')
            f.write(f'ID_PROMPT: {prompt["meta_data"]["id_prompt"]}')
            if "prompt" in prompt["request"]:
                f.write(prompt["request"]["prompt"])
            else: 
                f.write(prompt["request"])
            f.write("-" * 50)
            f.write("\n\n")

    for batch in tqdm(batched_inputs):
        ids = [el['meta_data']['id_prompt'] for el in batch]
        modes = [el['meta_data']['mode_prompt'] for el in batch]
        input_batch = [el["request"] for el in batch]
        gold_answers = [el['meta_data']['answer'] for el in batch]

        with time_counter(args.time_generate):
            outputs = llm.generate(
                input_batch,
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False
            )

        print("-" * 50)
        for id_out, o in enumerate(outputs):
            generated_text = o.outputs[0].text
            print("ID Question:", ids[id_out])
            print(generated_text)
            print("-" * 50)

            if "qwen3" in args.model_type.lower():
                thinking = generated_text.split("</think>")[0].strip() if "</think>" in generated_text else ""
                final_answer = generated_text.split("</think>")[1] if "</think>" in generated_text else ""
                final_answer = final_answer.split("Final Answer:")[1].strip() if "Final Answer:" in final_answer else ""

            else:
                thinking = generated_text.split("Final Answer:")[0].strip() if "Final Answer:" in generated_text else ""
                final_answer = generated_text.split("Final Answer:")[1].strip() if "Final Answer:" in generated_text else ""
            
            
            extracted_final_answer = extract_final_answer("Final Answer: " + final_answer.replace("*","").strip(), mode=modes[id_out])
            
            result = {"id_question": ids[id_out], "mode": modes[id_out], "gold_answer": gold_answers[id_out], "final_answer": extracted_final_answer, "correct": gold_answers[id_out] == extracted_final_answer, "completion": generated_text, "thinking_length": len(tokenizer.encode(thinking))}
            # # save results
            os.makedirs(f'out/completions/{args.model_type}/{args.subset}', exist_ok=True)
            with open(f"out/completions/{args.model_type}/{args.subset}/generations_{modes[id_out]}.jsonl", "a") as f:
                json.dump(result, f)
                f.write("\n")



if __name__ == "__main__":
    args = parse_args()
    main(args)