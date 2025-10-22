
import json
import copy
import argparse
import os


def get_templates(input_template_file_path):
    """Load templates from JSON file."""
    with open(input_template_file_path, "r") as f:
        return json.load(f)
    
def format_mcq_question(question_text, options):
    """Format a multiple choice question with options."""
    if "E" in options:
        return f"{question_text.strip()}\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n(D) {options['D']}\n(E) {options['E'].strip()}"
    return f"{question_text.strip()}\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n(D) {options['D'].strip()}"

def format_no_symbols_question(question_text, options):
    """Format a question without letter symbols."""
    return f"{question_text.strip()}\n- {options['A']}\n- {options['B']}\n- {options['C']}\n- {options['D'].strip()}"

def process_mcq_mode(item, templates=None):
    """Process MCQ mode."""
    new_item = copy.deepcopy(item)
    question = format_mcq_question(item['question'], item['options'])
    prompt = templates['mcq'].replace('<QUESTION>', question)
    return new_item, prompt

def process_open_mode(item, templates=None, open_questions=None):
    """Process open-ended mode."""
    new_item = copy.deepcopy(item)
    #new_item['options'] = {}  # Remove options for open-ended questions
    
    # Use manually re-adapted question if available
    if open_questions and item.get('id') in open_questions:
        question_text = open_questions[item['id']]['open_question']
        new_item['question'] = question_text
        # Also update the answer if provided in open dataset
        if 'open_answer' in open_questions[item['id']]:
            new_item['answer'] = open_questions[item['id']]['open_answer']
    else:
        question_text = item['open_question']
        new_item['answer'] = item['options'][item['answer']]  # Convert answer to text
    
    prompt = templates['open'].replace('<QUESTION>', question_text)
    return new_item, prompt

def process_incorrect_mode(item, templates=None, options_only_mode=False):
    """Process incorrect answers mode."""
    new_item = copy.deepcopy(item)
    question = format_mcq_question(item['question'], item['options'])
    # Set answer to list of incorrect options
    new_item['answer'] = [k for k in ['A', 'B', 'C', 'D'] if k != item['answer']]
    template_key = 'options_only_incorrect' if options_only_mode else 'incorrect'
    prompt = templates[template_key].replace('<QUESTION>', question)
    return new_item, prompt

def process_options_only_mode(item, templates=None, subset="medqa",  no_symbols=False, mode=""):
    """Process options-only mode."""
    if subset == "mmlu":
        subset = "MMLU"  # Use generic MMLU template
    elif subset == "medqa":
        subset = "MedQA USMLE"
    elif subset == "medmcqa":
        subset = "MedMCQA"
    new_item = copy.deepcopy(item)
    if mode == "no_symbols":
        options_text = "\n".join([f"- {v.strip()}" for k, v in item['options'].items()])
    else:
        options_text = "\n".join([f"({k}) {v.strip()}" for k, v in item['options'].items()])
    
    if not mode:
        prompt = templates['options_only'].replace("<OPTIONS>", options_text).replace("<DATASET>", subset)
    else:
        prompt = templates[f'options_only_{mode}'].replace("<OPTIONS>", options_text).replace("<DATASET>", subset)


    return new_item, prompt

def process_roman_numeral_mode(item, templates=None, options_only_mode=False):
    """Process roman numeral mode."""
    new_item = copy.deepcopy(item)
    id2roman = {'A': 'I', 'B': 'II', 'C': 'III', 'D': 'IV'}
    
    # Convert options and answer to roman numerals
    new_item['options'] = {id2roman[k]: v for k, v in item['options'].items()}
    new_item['answer'] = id2roman[item['answer']]
    
    # Format question with roman numerals
    question = f"{item['question'].strip()}\n(I) {new_item['options']['I']}\n(II) {new_item['options']['II']}\n(III) {new_item['options']['III']}\n(IV) {new_item['options']['IV'].strip()}"
    template_key = 'options_only_roman_numeral' if options_only_mode else 'roman_numeral'
    prompt = templates[template_key].replace('<QUESTION>', question)
    return new_item, prompt

def process_idk_answer_mode(item, templates=None, options_only_mode=False):
    """Process 'I don't know' mode."""
    new_item = copy.deepcopy(item)
    new_item['options']['E'] = "I don't know"
    question = format_mcq_question(item['question'], new_item['options'])
    template_key = 'options_only_idk_answer' if options_only_mode else 'idk_answer'
    prompt = templates[template_key].replace('<QUESTION>', question)
    return new_item, prompt

def process_yes_no_maybe_mode(item, templates=None):
    """Process yes/no/maybe mode."""
    new_item = copy.deepcopy(item)
    candidate_answer = item['options'][item['answer']]
    prompt = templates['yes_no_maybe'].replace('<QUESTION>', item['question'].strip()).replace('<CANDIDATE_ANSWER>', candidate_answer)
    return new_item, prompt

def process_none_of_provided_mode(item, templates=None, options_only_mode=False):
    """Process 'none of the provided' mode."""
    new_item = copy.deepcopy(item)
    gold_answer_key = item['answer']
    
    # Replace the correct answer with "None of the provided options"
    new_item['options'][gold_answer_key] = "None of the provided options"
    question = format_mcq_question(item['question'], new_item['options'])  # Use original options for question formatting
    template_key = 'options_only_none_of_the_provided' if options_only_mode else 'none_of_the_provided'
    prompt = templates[template_key].replace('<QUESTION>', question)
    return new_item, prompt

def process_fixed_pos_mode(item, templates=None, options_only_mode=False):
    """Process fixed position mode (correct answer always in position D)."""
    new_item = copy.deepcopy(item)
    
    # Find the correct answer text
    gold_answer_text = item['options'][item['answer']]
    
    # If gold answer is not already in position D, swap it
    if item['answer'] != 'D':
        # Find which position has the gold answer and swap with D
        for k, v in new_item['options'].items():
            if v == gold_answer_text:
                new_item['options'][k], new_item['options']['D'] = new_item['options']['D'], new_item['options'][k]
                break
    
    new_item['answer'] = 'D'
    question = format_mcq_question(item['question'], new_item['options'])
    template_key = 'options_only_fixed_pos' if options_only_mode else 'fixed_pos'
    prompt = templates[template_key].replace('<QUESTION>', question)
    return new_item, prompt

def process_no_symbols_mode(item, templates=None, options_only_mode=False):
    """Process no symbols mode."""
    new_item = copy.deepcopy(item)
    new_item['answer'] = item['options'][item['answer']]  # Answer becomes the text, not the letter
    question = format_no_symbols_question(item['question'], item['options'])
    template_key = 'options_only_no_symbols' if options_only_mode else 'no_symbols'
    prompt = templates[template_key].replace('<QUESTION>', question)
    return new_item, prompt

# Mode processing functions mapping
def get_mode_processors(open_questions=None, subset="medqa", templates=None, options_only_mode=False):
    """Get mode processors with optional open questions data."""
    
    if not options_only_mode:
        return {
            'mcq': lambda item: process_mcq_mode(item, templates),
            'open': lambda item: process_open_mode(item, templates, open_questions),
            'incorrect': lambda item: process_incorrect_mode(item, templates),
            'options_only': lambda item: process_options_only_mode(item, templates, subset),
            'roman_numeral': lambda item: process_roman_numeral_mode(item, templates),
            'none_of_the_provided': lambda item: process_none_of_provided_mode(item, templates),
            'fixed_pos': lambda item: process_fixed_pos_mode(item, templates),
            'no_symbols': lambda item: process_no_symbols_mode(item, templates),
        }
    else:
        return {
            'options_only_mcq': lambda item: process_options_only_mode(item, templates, subset, mode="mcq"),
            'options_only_incorrect': lambda item: process_options_only_mode(process_incorrect_mode(item, templates, options_only_mode=True)[0], subset, mode="incorrect"),
            'options_only_roman_numeral': lambda item: process_options_only_mode(process_roman_numeral_mode(item, templates, options_only_mode=True)[0], subset, mode="roman_numeral"),
            'options_only_none_of_the_provided': lambda item: process_options_only_mode(process_none_of_provided_mode(item, templates, options_only_mode=True)[0], subset, mode="none_of_the_provided"),
            'options_only_fixed_pos': lambda item: process_options_only_mode(process_fixed_pos_mode(item, templates, options_only_mode=True)[0], subset, mode="fixed_pos"),
            'options_only_no_symbols': lambda item: process_options_only_mode(process_no_symbols_mode(item, templates, options_only_mode=True)[0], subset, mode="no_symbols")
        }

def load_open_questions(open_file, subset="medqa"):
    """
    Load manually re-adapted open questions.
    
    Args:
        open_file (str): Path to JSON file with open questions
        
    Returns:
        dict: Dictionary mapping question IDs to open question data
    """
    if not open_file or not os.path.exists(open_file):
        return None
    
    try:
        with open(open_file, 'r') as f:
            data = json.load(f)[subset]
        
        # Handle different possible structures
        if isinstance(data, dict):
            # If it's a nested structure like {'dataset': {'id': {...}}}
            if len(data) == 1 and isinstance(list(data.values())[0], dict):
                return list(data.values())[0]
            # If it's already a flat structure like {'id': {...}}
            return data
        else:
            print(f"Warning: Unexpected structure in {open_file}")
            return None
            
    except Exception as e:
        print(f"Error loading open questions from {open_file}: {e}")
        return None

def process_benchmark(input_file, output_file, subset="medqa", templates=None, include_prompts=True, modes=None, open_file=None, options_only_mode=False):
    """
    Process benchmark data with different modes.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file  
        include_prompts (bool): Whether to include prompts in the output
        modes (list): List of specific modes to process (None for all)
        open_file (str): Path to JSON file with manually re-adapted open questions
    """
    # Load benchmark data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Load subset
    benchmark = data[subset]
    
    # Load open questions if provided
    open_questions = load_open_questions(open_file, subset)
    if open_questions:
        print(f"Loaded {len(open_questions)} manually re-adapted open questions")
    
    # Get mode processors
    mode_processors = get_mode_processors(open_questions, subset, templates, options_only_mode=options_only_mode)
    
    # Determine which modes to process
    all_modes = list(mode_processors.keys())
    if modes is None:
        modes_to_process = all_modes
    else:
        modes_to_process = [mode for mode in modes if mode in all_modes]
        invalid_modes = [mode for mode in modes if mode not in all_modes]
        if invalid_modes:
            print(f"Warning: Invalid modes ignored: {invalid_modes}")
    
    new_benchmark = {}
    
    # Process each mode
    for mode in modes_to_process:
        print(f"Processing mode: {mode}")
        new_benchmark[mode] = {}
        
        for question_id, item in benchmark.items():
            # try:
                # Ensure item has an ID for open questions lookup
                if 'id' not in item:
                    item['id'] = question_id
                
                # Process the item for this mode
                processor = mode_processors[mode]
                new_item, prompt = processor(item)
                
                # Optionally add prompt
                if include_prompts:
                    new_item['prompt'] = prompt
                
                new_benchmark[mode][question_id] = new_item
            
            # except Exception as e:
            #     print(f"Error processing question {question_id} in mode {mode}: {e}")
            #     continue
    
    # Save processed benchmark
    with open(output_file, 'w') as f:
        json.dump(new_benchmark, f, indent=2)
    
    print(f"Processing complete. Output saved to {output_file}")
    print(f"Processed modes: {modes_to_process}")

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Process medical benchmark data with different question formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all modes with prompts
  python script.py -i data/benchmark.json -o data/output.json --subset medqa
  
  # Process specific modes without prompts
  python script.py -i data/benchmark.json -o data/output.json --no-prompts -m mcq open incorrect
  
  # Process with manually re-adapted open questions
  python script.py -i data/benchmark.json -o data/output.json --open-file data/open_questions.json
  
  # Process only open mode with custom questions
  python script.py -i data/benchmark.json -o data/output.json -m open --open-file data/open_questions.json
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input JSON file containing benchmark data')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to output JSON file')
    parser.add_argument('-t', '--template', required=True,
                      help='Path to template input JSON file')
    
    # Optional arguments
    parser.add_argument('--no-prompts', action='store_true',
                      help='Do not include prompts in the output (default: include prompts)')
    parser.add_argument('-m', '--modes', nargs='+',
                      choices=['mcq', 'open', 'incorrect', 'options_only', 'roman_numeral', 
                              'yes_no_maybe', 'none_of_the_provided', 'fixed_pos', 'no_symbols'],
                      help='Specific modes to process (default: all modes)')
    parser.add_argument('--open-file', type=str,
                      help='Path to JSON file with manually re-adapted open questions')
    parser.add_argument('--subset', type=str, choices=['medqa', 'medmcqa', 'mmlu'], default='mmlu',
                      help='Dataset subset to use (default: medqa)')
    parser.add_argument('--list-modes', action='store_true',
                      help='List available modes and exit')
    parser.add_argument('--options-only-mode', action='store_true',
                    help="Activate prompt strategy for 'options only' mode.")

    
    args = parser.parse_args()
    
    # List modes if requested
    if args.list_modes:
        print("Available modes:")
        modes = ['mcq', 'open', 'incorrect', 'options_only', 'roman_numeral', 
                'yes_no_maybe', 'none_of_the_provided', 'fixed_pos', 'no_symbols', 'idk_answer']
        for mode in modes:
            print(f"  - {mode}")
        return
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # Validate open file if provided
    if args.open_file and not os.path.exists(args.open_file):
        print(f"Warning: Open questions file '{args.open_file}' does not exist")
        print("Continuing without open questions...")
        args.open_file = None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process benchmark
    include_prompts = not args.no_prompts
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Include prompts: {include_prompts}")
    print(f"Modes: {args.modes if args.modes else 'all'}")
    if args.open_file:
        print(f"Open questions file: {args.open_file}")
    print("-" * 50)
    
    # try:
    process_benchmark(
        input_file=args.input,
        output_file=args.output,
        subset=args.subset,
        include_prompts=include_prompts,
        modes=args.modes,
        open_file=args.open_file, 
        templates=get_templates(args.template),
        options_only_mode=args.options_only_mode
    )
    # except Exception as e:
    #     print(f"Error during processing: {e}")
    #     return
    
    print("-" * 50)
    print("Done!")

if __name__ == "__main__":
    main()