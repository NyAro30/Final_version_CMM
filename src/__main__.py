#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   __main__.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/20 19:45:29 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/20 22:57:12 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""Entry point for the Call me maybe function calling tool."""

import argparse
import json
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model

from src.schema import load_functions, generate_function_call
from src.vocab import load_vocab


def main() -> None:
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

    model = Small_LLM_Model(device="cpu")
    vocab = load_vocab(model)

    with open(args.input, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results: list[dict[str, Any]] = []
    for prompt_data in prompts:
        user_request = prompt_data["prompt"]
        print(f"Processing: {user_request}")

        call = generate_function_call(
            model, vocab, functions, user_request
        )

        results.append({
            "prompt": user_request,
            "name": call["name"],
            "parameters": call["parameters"],
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
