#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_constrained_generation.py                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/17 08:34:38 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/17 10:05:48 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from llm_sdk.llm_sdk import Small_LLM_Model
from test_vocab import load_vocab
from test_json_validator import is_valid_json, is_valid_json_prefix
from test_valid_tokens import valid_next_tokens


def generate_constrained(model: Small_LLM_Model, vocab: dict[int, str],
                         prompt: str, max_tokens: int = 50) -> str:
    """Generate JSON using constrained decoding.
    
    At each step:
    1. Get logits from model.
    2. Determine valid next tokens (based on JSON prefix so far).
    3. Mask invalid tokens (set their logit to -inf).
    4. Pick argmax of remaining.
    5. Append to sequence.
    6. Stop when JSON is complete.
    """

    prompts_id =  model.encode(prompt)
    ids = ids = prompts_id.tolist()[0]
    
    generated_ids: list[int] = []
    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids)
        json_prefix = model.decode(generated_ids)
        valid = valid_next_tokens(json_prefix,vocab)
        if valid == []:
            break
        valid_test = set(valid)
        masked = [logits[i] if i in valid_test else float('-inf') for i in range(len(logits))]
        next_id = masked.index(max(masked))
        ids.append(next_id)
        generated_ids.append(next_id)
        print(f"Step {i + 1}: token_id={next_id}, decoded={model.decode([next_id])!r}")
        if is_valid_json(model.decode(generated_ids)):
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
