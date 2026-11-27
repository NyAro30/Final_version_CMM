#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   schema.py                                            :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/26 13:10:41 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/28 01:19:45 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""Function selection, argument extraction, and constrained JSON values."""

from __future__ import annotations

import json
import math
import os
from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError

from .models import FunctionDef, LLMModel
from .json_prefix import (
    is_valid_boolean_prefix,
    is_valid_number_prefix,
    is_valid_string_prefix,
    scalar_before_separator,
)
from .prompts import arguments_prompt, chat_prompt
from .validator import is_valid_json

VERBOSE = os.environ.get("VERBOSE", "") == "1"


def _ranked_tokens(logits: list[float], vocab: dict[int, str]) -> list[int]:
    """Return valid vocabulary IDs from the highest logit to the lowest."""
    ids = [token for token in vocab if 0 <= token < len(logits)]
    return sorted(ids, key=logits.__getitem__, reverse=True)


def generate_parameter_value(
    model: LLMModel,
    vocab: dict[int, str],
    prompt: str,
    param_type: str,
    max_tokens: int = 128,
    assistant_prefix: str = "",
) -> str:

    """Generate one JSON scalar while masking tokens that violate its type."""

    if param_type == "integer":
        param_type = "number"

    validators = {
        "number": is_valid_number_prefix,
        "string": is_valid_string_prefix,
        "boolean": is_valid_boolean_prefix,
    }
    if param_type not in validators:
        raise ValueError(f"Unsupported parameter type: {param_type!r}")

    ids = model.encode(
        chat_prompt(
            "Fill the function arguments from the request.",
            prompt,
        ) + assistant_prefix
    ).tolist()[0]
    generated: list[int] = []

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        prefix = model.decode(generated)
        ranked = _ranked_tokens(logits, vocab)
        if param_type == "number" and generated and ranked:
            try:
                complete_number = json.loads(prefix)
            except JSONDecodeError:
                complete_number = None
            if (
                isinstance(complete_number, (int, float))
                and not isinstance(complete_number, bool)
            ):
                next_prefix = model.decode(generated + [ranked[0]])
                if (
                    next_prefix == prefix
                    or not is_valid_number_prefix(next_prefix)
                ):
                    break
        token = next((
            token for token in ranked
            if (
                (candidate := model.decode(generated + [token])) != prefix
                and (
                    validators[param_type](candidate)
                    or scalar_before_separator(candidate, param_type)
                    is not None
                )
            )
        ), None)
        if token is None:
            if param_type == "number":
                try:
                    value = json.loads(prefix)
                except JSONDecodeError:
                    value = None
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    break
            raise ValueError(f"No valid token for {param_type} value")

        generated.append(token)
        ids.append(token)
        prefix = model.decode(generated)

        if VERBOSE:
            print(f"Step {step + 1}: {model.decode([token])!r}")
        if scalar_before_separator(prefix, param_type) is not None:
            break
        if param_type == "string":
            try:
                if isinstance(json.loads(prefix), str):
                    break
            except JSONDecodeError:
                pass
        elif param_type == "boolean" and prefix in ("true", "false"):
            break
        elif param_type == "number":
            try:
                json.loads(prefix)
            except JSONDecodeError:
                continue
            greedy = _ranked_tokens(logits, vocab)
            if not greedy or not validators[param_type](
                model.decode(generated + [greedy[0]])
            ):
                break

    generated_text = model.decode(generated)
    result = (
        scalar_before_separator(generated_text, param_type)
        or generated_text
    )
    if not is_valid_json(result):
        raise ValueError(f"Generated invalid JSON value: {result!r}")
    try:
        value = json.loads(result)
    except JSONDecodeError as err:
        raise ValueError(f"Generated invalid JSON value: {result!r}") from err
    expected_type = {
        "number": (int, float),
        "string": (str,),
        "boolean": (bool,),
    }[param_type]
    if isinstance(value, bool) and param_type == "number":
        raise ValueError("Generated boolean where a number was required")
    if not isinstance(value, expected_type):
        raise ValueError(f"Generated value does not match {param_type}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Generated a non-finite number")
    return result


def generate_function_name(
    model: LLMModel,
    vocab: dict[int, str],
    prompt: str,
    function_names: list[str],
    max_tokens: int = 20,
) -> str:

    """Use model logits to select a valid function name token by token."""

    ids = model.encode(prompt).tolist()[0]
    generated: list[int] = []
    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        prefix = model.decode(generated)
        token = next((
            token for token in _ranked_tokens(logits, vocab)
            if (candidate := model.decode(generated + [token])) != prefix
            and any(name.startswith(candidate) for name in function_names)
        ), None)
        if token is None:
            break
        generated.append(token)
        ids.append(token)
        name = model.decode(generated)
        if VERBOSE:
            print(f"Name step {step + 1}: {name!r}")
        if name in function_names:
            return name
    return model.decode(generated)


def load_functions(path: str) -> dict[str, dict[str, Any]]:

    """Load and validate function definitions from a JSON array."""

    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("function definitions must be a JSON array")
    except (OSError, JSONDecodeError, ValueError) as err:
        print(f"Error loading functions: {err}")
        return {}

    functions: dict[str, dict[str, Any]] = {}
    for item in data:
        try:
            definition = FunctionDef.model_validate(item)
            if definition.name in functions:
                raise ValueError(f"duplicate function name: {definition.name}")
            functions[definition.name] = definition.model_dump()
        except (ValidationError, ValueError) as err:
            print(f"Skipping invalid function definition: {err}")
    return functions


def generate_function_call(
    model: LLMModel,
    vocab: dict[int, str],
    functions: dict[str, dict[str, Any]],
    user_request: str,
) -> dict[str, Any]:

    """Select a function with the LLM and build validated typed arguments."""

    if not functions:
        raise ValueError("No function definitions were provided")

    choices = "\n".join(
        f"- {name}: {definition['description']}"
        for name, definition in functions.items()
    )
    prompt = chat_prompt(
        "Choose the best matching function. Return only its exact name.",
        f"Available functions:\n{choices}\nRequest: {user_request}",
    )
    name = generate_function_name(model, vocab, prompt, list(functions))
    if name not in functions:
        raise ValueError(f"Model selected invalid function name: {name!r}")

    definition = functions[name]
    parameters: dict[str, Any] = {}
    parameter_types = {
        parameter_name: parameter_info["type"]
        for parameter_name, parameter_info in definition.get(
            "parameters", {}
        ).items()
    }
    value_prompt = arguments_prompt(
        name,
        definition["description"],
        user_request,
        parameter_types,
    )
    object_prefix = "{"
    for param_name, param_info in definition.get("parameters", {}).items():
        original_type = param_info["type"]
        gen_type = "number" if original_type in ("integer", "float") else original_type

        if parameters:
            object_prefix += ", "
        object_prefix += f"{json.dumps(param_name)}: "
        value = json.loads(
            generate_parameter_value(
                model,
                vocab,
                value_prompt,
                gen_type,
                assistant_prefix=object_prefix,
            )
        )
        if gen_type == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ValueError(f"Invalid number argument {param_name!r}")
        
        if original_type == "integer":
            value = int(value)
        elif original_type == "number":
            value = float(value)
        
        if original_type == "string" and not isinstance(value, str):
            raise ValueError(f"Invalid string argument {param_name!r}")
        if original_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"Invalid boolean argument {param_name!r}")
        parameters[param_name] = value
        object_prefix += json.dumps(value, ensure_ascii=False)
    return {"name": name, "parameters": parameters}
