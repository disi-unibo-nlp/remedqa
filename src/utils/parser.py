import re 

def parse_mcq_answer(response: str):
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

def parse_mcq_answer_roman(response: str, convert_roman=False):
    """
    Extracts the MCQ letter from a response containing 'Final Answer: (X)'.
    
    Handles cases like:
    - 'Final Answer: (I)'
    - 'Final Answer: I'
    - 'Answer: (II)'
    - 'Final Answer: III'
    Returns the option letter as a string (e.g., 'A'), or None if not found.
    """
    if not convert_roman:
        
        match = re.search(r"(?:Final Answer:|Answer:)\s*\(?([IVX]+)\)?", response, re.IGNORECASE)
        if match:
            roman_numeral = match.group(1).upper()
            return roman_numeral
    else:
        match = re.search(r"(?:Final Answer:|Answer:)\s*\(?([A-D])\)?", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None
   

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

def parse_incorrect_answer(response: str):
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

def parse_final_answer_standard(response: str, mode: str):
    if mode in ['mcq', 'options_only', 'none_of_the_provided', 'fixed_pos', 'idk_answer']:
        return parse_mcq_answer(response)
    elif mode == "roman_numeral":
        return parse_mcq_answer_roman(response)
    elif mode == "incorrect":
        return parse_incorrect_answer(response)
    else:
        return parse_open_answer(response)


def parse_mcq_answer_think(text: str) -> str | None:
    """
    Extracts the answer letter from strings like:
      - 'The final answer is \\boxed{C}'
      - 'Final Answer: (C)'
      - '\\boxed{\\text{A}}'
      - '$\\boxed{\\text{B}}$'
    Returns a single letter A–E if found, else None.
    """
    # Try LaTeX boxed format with optional \text{} and optional $
    match = re.search(r"\$?\\boxed\{(?:\\text\{)?([A-E])\}?\}\$?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try parentheses format: (C)
    match = re.search(r"\(([A-E])\)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None

import re

def parse_open_answer_think(response: str) -> str | None:
    """
    Extracts the open-ended answer from a response containing 'Final Answer: <answer>'
    or wrapped in LaTeX \boxed{...} or \boxed{\text{...}}.
    
    Handles cases like:
      - 'Final Answer: The answer is 42.'
      - 'Answer: 42'
      - 'Answer: The capital of France is Paris.'
      - '$\\boxed{Common iliac artery aneurysm}$'
      - '$\\boxed{\\text{Common iliac artery aneurysm}}$'
    Returns the answer as a string, or None if not found.
    """
    # Step 1: Strip optional boxed wrapper
    boxed_match = re.search(r"\$?\\boxed\{(?:\\text\{)?(.+?)(?:\})?\}\$?", response, re.IGNORECASE)
    if boxed_match:
        response = boxed_match.group(1).strip()

    # Step 2: Strip optional 'Final Answer:' or 'Answer:' prefix
    match = re.search(r"(?:Final Answer:|Answer:)\s*(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Step 3: If no prefix, return the text as-is (after box removal)
    if response:
        return response.strip()

    return None



def parse_mcq_answer_roman_think(text: str) -> str | None:
    """
    Extracts the Roman numeral answer from strings like:
      - '\\boxed{III}'
      - '\\boxed{\\text{III}}'
      - '$\\boxed{\\text{III}}$'
      - '$\\boxed{\\text{(III)}}$'
      - '(IV)'
    Works for I, II, III, IV, V, etc.
    Returns the matched Roman numeral as a string or None if not found.
    """
    # Primary: boxed versions, with optional \text and parentheses
    match = re.search(
        r"\$?\\boxed\{(?:\\text\{)?\(?([IVXLCDM]+)\)?\}?\}\$?",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # Fallback: just (IV), (X), etc.
    fallback = re.search(r"\(([IVXLCDM]+)\)", text, re.IGNORECASE)
    if fallback:
        return fallback.group(1).upper()

    return None


def parse_incorrect_answer_think(text: str):
    """
    Extract answers from patterns like:
      - \boxed{A,B,D}
      - $\boxed{\text{A,B,D}}$
      - A,B,D   or   A, B, D   (plain text)
      - ABD
      - (A), (B), (D)
      - (A),(B),(D)
    Returns a list of stripped answers if found, else None.
    """
    # Try boxed with optional \text and $ delimiters
    match = re.search(r"\$?\\boxed\{(?:\\text\{)?(.+?)(?:\})?\}\$?", text)
    if match:
        content = match.group(1)
    else:
        # Try plain comma-separated letters
        plain_match = re.search(r"\b([A-D](?:\s*,\s*[A-D]){1,3})\b", text)
        if plain_match:
            content = plain_match.group(1)
        else:
            # Try concatenated letters like "ABD"
            concat_match = re.search(r"\b([A-D]{2,4})\b", text)
            if concat_match:
                return list(concat_match.group(1))
            # Try parenthesized form like "(A), (B), (D)" or "(A),(B),(D)"
            paren_matches = re.findall(r"\(([A-D])\)", text)
            if paren_matches:
                return paren_matches
            return None

    # Default split by commas and strip whitespace
    answers = [ans.strip() for ans in content.split(",")]
    return answers



def parse_yes_no_idk(text: str) -> str | None:
    """
    Extracts an answer ('Yes', 'No', or 'I don't know')
    from strings like:
      - 'Final Answer: Yes'
      - 'Yes'
      - 'No'
      - 'I don't know'
      - '\\boxed{Yes}'
      - '\\boxed{\\text{Yes}}'
    Returns the matched answer as a string or None if not found.
    """
    # Strip outer \boxed{...} and optional \text{...}
    boxed_match = re.search(r"\\boxed\{(?:\\text\{)?(.+?)(?:\})?\}", text, re.IGNORECASE)
    if boxed_match:
        text = boxed_match.group(1)

    text_lower = text.lower().strip()

    # Remove surrounding quotes if present
    text_lower = text_lower.strip('"').strip("'")

    # Remove optional 'final answer:' prefix
    text_lower = re.sub(r'^final\s*answer\s*:\s*', '', text_lower)

    # Normalize spacing (for "i don't know")
    text_lower = re.sub(r"\s+", " ", text_lower)

    # Match allowed answers
    if text_lower.startswith("yes"):
        return "Yes"
    elif text_lower.startswith("no"):
        return "No"
    elif text_lower.startswith("i don't know") or text_lower.startswith("i do not know"):
        return "I don't know"
    else:
        return None


def parse_final_answer_think(response: str, mode: str):
    if not response:
        return ""
    if mode in ['mcq', 'options_only', 'none_of_the_provided', 'fixed_pos']:
        return parse_mcq_answer_think(response)
    elif mode == "roman_numeral":
        return parse_mcq_answer_roman_think(response)
    elif mode == "incorrect":
        return parse_incorrect_answer_think(response)
    elif mode == "yes_no_maybe":
        return parse_yes_no_idk(response)
    else:
        if "answer:" not in response.lower():
            response = "Final Answer: " + response
        return parse_open_answer_think(response)

def parse_final_answer(response: str, mode: str, model_name: str):
    if "openai" in model_name.lower() or "gemini" in model_name.lower() or "gpt-oss" in model_name.lower():
        return parse_final_answer_think(response=response, mode=mode)
    else:
        return parse_final_answer_standard(response=response, mode=mode)