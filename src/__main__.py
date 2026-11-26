#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   __main__.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/20 19:45:29 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/26 16:23:28 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""The architect that parse the arguments ,
   process the prompts, and write the output"""

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import InputPrompt, OutputResult
from .schema import generate_function_call, load_functions
from .vocab import load_vocab


def main() -> int:

    """Main entry point: parse args, process prompts, write output."""

    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output JSON file.",
    )

    args = parser.parse_args()

    functions = load_functions(args.functions_definition)
    if not functions:
        print("Error: no valid function definitions were loaded")
        return 1

    try:
        with open(args.input, "r", encoding="utf-8") as input_file:
            raw_prompts = json.load(input_file)
    except (OSError, json.JSONDecodeError) as err:
        print(f"Error reading prompts from {args.input}: {err}")
        return 1

    if not isinstance(raw_prompts, list):
        print(
            f"Error reading prompts from {args.input}: "
            "expected a JSON array"
        )
        return 1

    prompts: list[InputPrompt] = []
    input_errors = 0
    for index, item in enumerate(raw_prompts):
        try:
            prompts.append(InputPrompt.model_validate(item))
        except ValidationError as err:
            print(f"Skipping invalid prompt at index {index}: {err}")
            input_errors += 1

    if not prompts:
        status = _write_results(args.output, [])
        return status or int(input_errors > 0)

    try:
        from llm_sdk.llm_sdk import Small_LLM_Model

        model = Small_LLM_Model(device="cpu")
        vocab = load_vocab(model)
    except Exception as err:
        print(f"Error loading Qwen model or vocabulary: {err}")
        return 1

    results: list[dict[str, Any]] = []
    failed_prompts = input_errors
    for prompt_data in prompts:
        try:
            call = generate_function_call(
                model, vocab, functions, prompt_data.prompt
            )
            result = OutputResult(
                prompt=prompt_data.prompt,
                name=call["name"],
                parameters=call["parameters"],
            )
            results.append(result.model_dump())
        except Exception as err:
            print(f"Error processing prompt {prompt_data.prompt!r}: {err}")
            failed_prompts += 1

    status = _write_results(args.output, results)
    if status:
        return status
    if failed_prompts:
        print(
            f"Wrote {len(results)} successful result(s); "
            f"{failed_prompts} prompt(s) failed."
        )
        return 1
    return 0


def _write_results(path: str, results: list[dict[str, Any]]) -> int:

    """Write all generated results as strict JSON."""

    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, allow_nan=False)
    except (OSError, ValueError) as err:
        print(f"Error writing output to {path}: {err}")
        return 1

    print(f"Output written to {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        raise SystemExit(130)
