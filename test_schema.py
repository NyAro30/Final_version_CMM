#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_schema.py                                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/18 14:47:30 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/20 10:18:35 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from json import JSONDecodeError
from llm_sdk.llm_sdk import Small_LLM_Model
from test_vocab import load_vocab


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
