#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_schema.py                                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/18 14:47:30 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/18 16:02:57 by mny-aro-           ###   ########.fr      #
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

    At each step, only tokens that could continue a valid function name are allowed.

    Args:
        model: The LLM model instance.
        vocab: Dict {token_id: token_string}.
        prompt: Instruction prompt for the model.
        function_names: List of valid function names.
        max_tokens: Max tokens to generate.

    Returns:
        The generated function name (must be one of function_names).
    """

    prompt_ids = model.encode(prompt)
    generated_ids = prompt_ids.tolist()[0]
    
    name_prefix = ""

    i = 0
    for i in range(max_tokens):
        



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

    print("Functions loaded:", len(functions))
    for name, fn_def in functions.items():
        params = fn_def.get("parameters", {})
        param_types = {k: v["type"] for k, v in params.items()}
        print(f"  {name}: params={param_types}")
