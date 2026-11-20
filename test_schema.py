#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_schema.py                                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/18 14:47:30 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/20 12:56:22 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from json import JSONDecodeError
from llm_sdk.llm_sdk import Small_LLM_Model
from test_vocab import load_vocab
from test_json_validator import is_valid_json


def is_valid_number_prefix(s: str) -> bool:
    """Return True if s could be a prefix of a valid JSON number."""
    if not s:
        return True
    # All chars must be valid number characters
    if not all(c in "0123456789.+-eE" for c in s):
        return False
    # Must start with digit or minus
    if s[0] not in "0123456789-":
        return False
    # Max one of each special char
    if s.count('-') > 1 or s.count('.') > 1:
        return False
    if s.lower().count('e') > 1:
        return False
    return True


def is_valid_string_prefix(s: str) -> bool:
    """Return True if s could be a prefix of a valid JSON string."""
    if not s:
        return True  # could start with "
    if not s.startswith('"'):
        return False
    # Try to complete and parse
    try:
        json.loads(s + '"')
        return True
    except json.JSONDecodeError:
        # Might be in the middle of an escape sequence
        if s.endswith('\\'):
            return True
        return False


def generate_parameter_value(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    param_type: str,
    max_tokens: int = 20
) -> str:
    """Generate a parameter value using the LLM, constrained to the right type.

    Args:
        model: The LLM model instance.
        vocab: Dict {token_id: token_string}.
        prompt: Instruction prompt for the model.
        param_type: "number" or "string".
        max_tokens: Max tokens to generate.

    Returns:
        The generated value as a string.
    """
    ids = model.encode(prompt).tolist()[0]
    generated_ids: list[int] = []

    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        value_prefix = model.decode(generated_ids)

        valid = []
        for token_id, token_str in vocab.items():
            clean_token = token_str.replace("Ġ", " ")
            new_prefix = value_prefix + clean_token

            if param_type == "number":
                if is_valid_number_prefix(new_prefix):
                    valid.append(token_id)
            elif param_type == "string":
                if is_valid_string_prefix(new_prefix):
                    valid.append(token_id)

        if not valid:
            break

        valid_set = set(valid)
        masked = [logits[j] if j in valid_set else float('-inf')
                  for j in range(len(logits))]
        next_id = masked.index(max(masked))

        ids.append(next_id)
        generated_ids.append(next_id)
        print(f"Step {i+1}: token_id={next_id},"
              f"decoded={model.decode([next_id])!r}")

        value_prefix = model.decode(generated_ids)

        if param_type == "number":
            if value_prefix and is_valid_json(value_prefix):
                break
        elif param_type == "string":
            if (value_prefix.startswith('"') and value_prefix.endswith('"') and
               not value_prefix.endswith('\\')):
                break

    return model.decode(generated_ids)


def generate_function_name(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    prompt: str,
    function_names: list[str],
    max_tokens: int = 20
) -> str:
    """Generate a function name using the LLM, constrained to valid names.

    At each step, only tokens that could continue a valid function name are
    allowed.

    Args:
        model: The LLM model instance.
        vocab: Dict {token_id: token_string}.
        prompt: Instruction prompt for the model.
        function_names: List of valid function names.
        max_tokens: Max tokens to generate.

    Returns:
        The generated function name (must be one of function_names).
    """

    ids = model.encode(prompt).tolist()[0]
    generated_ids: list[int] = []

    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        name_prefix = model.decode(generated_ids)

        valid = []
        for token_id, token_str in vocab.items():
            clean = token_str.replace("Ġ", " ")
            new_prefix = name_prefix + clean
            if any(fn.startswith(new_prefix) for fn in function_names):
                valid.append(token_id)

        if not valid:
            break

        valid_set = set(valid)
        masked = [logits[j] if j in valid_set else float('-inf')
                  for j in range(len(logits))]
        next_id = masked.index(max(masked))

        ids.append(next_id)
        generated_ids.append(next_id)
        print(f"Step {i+1}: token_id={next_id},"
              f"decoded={model.decode([next_id])!r}")

        name_prefix = model.decode(generated_ids)
        if name_prefix in function_names:
            break

    return model.decode(generated_ids)


def load_functions(path: str) -> dict[str, dict]:
    """Load function definitions from a JSON file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        Dict {function_name: function_definition}.
        Each function_def has: name, description, parameters, returns.
    """

    with open(path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except JSONDecodeError as err:
            print(f"Error :{err}")
            return {}
    functions_dict = {func["name"]: func for func in data}

    return functions_dict


if __name__ == "__main__":
    functions = load_functions("data/input/functions_definition.json")
    function_names = list(functions.keys())

    model = Small_LLM_Model(device="cpu")
    vocab = load_vocab(model)

    prompt = """Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- fn_reverse_string: Reverse a string and return the reversed result.
- fn_get_square_root: Calculate the square root of a number.
- fn_substitute_string_with_regex: Replace all occurrences matching a regex
pattern in a string.

User request: What is the sum of 2 and 3?

Select the most appropriate function name:"""

    name = generate_function_name(model, vocab, prompt, function_names)
    print("---")
    print("Selected function:", repr(name))
    print("Is valid:", name in function_names)
    print()
    print("=== Test generate_parameter_value ===")
    prompt_a = """Function: fn_add_numbers (Add two numbers together)
    User request: What is the sum of 2 and 3?
    Generate the value for parameter 'a' (a number):"""

    value_a = generate_parameter_value(model, vocab, prompt_a, "number")
    print("---")
    print("Value for 'a':", repr(value_a))

    prompt_b = """Function: fn_add_numbers (Add two numbers together)
    User request: What is the sum of 2 and 3?
    The value for parameter 'a' is 2.
    Generate the value for parameter 'b' (a number):"""

    value_b = generate_parameter_value(model, vocab, prompt_b, "number")
    print("Value for 'b':", repr(value_b))
