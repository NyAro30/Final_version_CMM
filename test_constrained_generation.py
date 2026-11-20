#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_constrained_generation.py                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/17 08:34:38 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/18 11:00:32 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from llm_sdk.llm_sdk import Small_LLM_Model
from test_vocab import load_vocab
from test_json_validator import is_valid_json_prefix, is_valid_json, is_valid_json_object, is_valid_json_object_prefix
from test_valid_tokens import valid_next_tokens


def generate_constrained(model: Small_LLM_Model, vocab: dict[int, str],
                         prompt: str, max_tokens: int = 50) -> str:
    """Generate JSON object using constrained decoding."""
    prompts_id = model.encode(prompt)
    ids = prompts_id.tolist()[0]
    generated_ids: list[int] = []

    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        json_prefix = model.decode(generated_ids)
        valid = valid_next_tokens(json_prefix, vocab)

        if not valid:
            print("No valid tokens! Stuck.")
            break

        valid_set = set(valid)
        masked = [logits[i] if i in valid_set
                  else float('-inf') for i in range(len(logits))]
        next_id = masked.index(max(masked))

        ids.append(next_id)
        generated_ids.append(next_id)
        print(f"Step {i + 1}: token_id={next_id},"
              f"decoded={model.decode([next_id])!r}")

        if is_valid_json_object(model.decode(generated_ids)):
            print("JSON object complete!")
            break

    return model.decode(generated_ids)


if __name__ == "__main__":
    model = Small_LLM_Model(device="cpu")
    vocab = load_vocab(model)

    prompt = "Generate a JSON function call for: What is the sum of 2 and 3?"
    result = generate_constrained(model, vocab, prompt, max_tokens=50)
    print("---")
    print("Result:", repr(result))
    print("Is valid JSON:", is_valid_json(result))
    print("Is valid JSON object:", is_valid_json_object(result))
