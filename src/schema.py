#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   schema.py                                            :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/20 19:46:10 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/25 00:57:01 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #


import json
from .models import FunctionDef
from json import JSONDecodeError
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any


def is_valid_number_prefix(s: str) -> bool:
    
    """Return True if s could be a prefix of a valid JSON number."""
    
    if not s:
        return True
    
    if not all(c in "0123456789.+-eE" for c in s):
        return False
    
    if s[0] not in "0123456789-":
        return False
    
    if s in ("+", "-", "e", "E", "+e", "-e", "."):
        return True

    if '-' in s[1:] and 'e' not in s[:s.index('-', 1)].lower():
        return False
    
    if s.count('-') > 1 or s.count('.') > 1:
        return False
    
    if s.lower().count('e') > 1:
        return False
    
    return True


def is_valid_string_prefix(s: str) -> bool:

    """Return True if s could be a prefix of a valid JSON string."""

    if not s:
        return True

    if not s.startswith('"'):
        return False

    i = 1
    in_escape = False

    while i < len(s):
        c = s[i]

        if in_escape:

            if c not in '"\\/bfnrtu':
                return False

            if c == 'u':
                hex_part = s[i + 1:i + 5]

                if len(hex_part) < 4:
                    return True

                if not all(h in '0123456789abcdefABCDEF' for h in hex_part):
                    return False
                i += 5

            else:
                i += 1
            in_escape = False

        elif c == '\\':
            in_escape = True
            i += 1

        elif c == '"':

            if i + 1 < len(s):
                return False

            return True
        elif ord(c) < 32:
            return False

        else:
            i += 1

    return True


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

        valid: list[int] = []
        for token_id, token_str in vocab.items():
            new_prefix = value_prefix + token_str

            if param_type == "number":
                if is_valid_number_prefix(new_prefix):
                    valid.append(token_id)
            elif param_type == "string":
                if is_valid_string_prefix(new_prefix):
                    valid.append(token_id)

        if not valid:
            break

        valid_set = set(valid)

        if param_type == "number" and generated_ids:
            unconstrained_argmax = logits.index(max(logits))
            if unconstrained_argmax not in valid_set:
                break

        masked = [
            logits[j] if j in valid_set else float('-inf')
            for j in range(len(logits))
        ]
        next_id = masked.index(max(masked))

        ids.append(next_id)
        generated_ids.append(next_id)
        print(
            f"Step {i + 1}: token_id={next_id},"
            f"decoded={model.decode([next_id])!r}"
        )

        if len(generated_ids) >= 6:
            for pattern_len in [2, 3, 4]:
                if len(generated_ids) >= pattern_len * 2:
                    last = generated_ids[-pattern_len:]
                    prev = generated_ids[-pattern_len * 2: -pattern_len]
                    if last == prev:
                        print(f"Loop detected (pattern {pattern_len},"
                              f" stopping.)")
                        break
            else:
                continue
            break

        if param_type == "string":
            value_prefix = model.decode(generated_ids)
            try:
                json.loads(value_prefix)
                break
            except json.JSONDecodeError:
                pass

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
            new_prefix = name_prefix + token_str
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


def load_functions(path: str) -> dict[str, dict[str, Any]]:

    """Load function definitions from a JSON file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        Dict {function_name: function_definition}.
        Each function_def has: name, description, parameters, returns.
    """

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (JSONDecodeError, FileNotFoundError) as err:
        print(f"Error loading functions: {err}")
        return {}

    functions_dict: dict[str, dict[str, Any]] = {}
    for func in data:
        try:
            validated = FunctionDef(**func)
            functions_dict[validated.name] = validated.model_dump()
        except Exception as err:
            print(f"Invalid function definition: {err}")
    return functions_dict


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    decoded: dict[int, str] = {}
    total = len(raw_vocab)
    for i, (token_str, token_id) in enumerate(raw_vocab.items()):
        decoded[token_id] = model.decode([token_id])
        if i % 10000 == 0:
            print(f"Decoding vocab: {i}/{total}")
    return decoded


def generate_function_call(
    model: Small_LLM_Model,
    vocab: dict[int, str],
    functions: dict[str, dict[str, Any]],
    user_request: str,
    max_tokens: int = 20
) -> dict[str, Any]:

    """Generate a complete function call using constrained decoding.

    1. Select function name (constrained to valid names).
    2. For each parameter, generate value (constrained to type).
    3. Assemble into {"name": ..., "parameters": {...}}.

    Returns:
        Dict with "name" and "parameters" keys.
    """

    function_names = list(functions.keys())

    prompt_fn = ""
    for fn_name, fn_def in functions.items():
        desc: str = fn_def["description"]
        prompt_fn += f"- {fn_name}: {desc}\n"
    prompt_fn += (
        f"\nUser request: {user_request}\n\n"
        "Select the most appropriate function name:"
    )

    name = generate_function_name(model, vocab, prompt_fn, function_names)

    if name not in functions:
        print(f"Error: invalid function name '{name}'")
        return {"name": "", "parameters": {}}

    fn_def = functions[name]
    params: dict[str, Any] = fn_def.get("parameters", {})

    parameters: dict[str, Any] = {}
    for param_name, param_info in params.items():
        param_type: str = param_info["type"]

        prompt_param = (
            f"Function: {name} ({fn_def['description']})\n"
            f"    User request: {user_request}\n"
        )

        for prev_name, prev_value in parameters.items():
            prompt_param += (
                f"    The value for parameter '{prev_name}' is {prev_value}.\n"
            )

        prompt_param += (
            f"    Generate the value for parameter "
            f"'{param_name}' (a {param_type}):"
        )

        value_str = generate_parameter_value(
            model, vocab, prompt_param, param_type
        )

        try:
            value: Any = json.loads(value_str)
        except json.JSONDecodeError:
            print(f"Warning: could not parse value {value_str!r}")

            if param_type == "number":
                value = 0
            else:
                value = value_str
        parameters[param_name] = value
    return {"name": name, "parameters": parameters}
