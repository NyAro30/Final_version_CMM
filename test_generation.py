#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_generation.py                                   :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/07 21:29:27 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/07 23:46:44 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from llm_sdk.llm_sdk import Small_LLM_Model


def generate(model: Small_LLM_Model, prompt: str, max_tokens: int = 20) -> str:
    """Generate text by repeatedly predicting the next token.
    
    Args:
        model: The LLM model instance.
        prompt: The input prompt.
        max_tokens: Maximum number of tokens to generate.
    
    Returns:
        The full text (prompt + generated tokens).
    """
    ids_tensor = model.encode(prompt)
    
    ids_list = ids_tensor.tolist()[0]

    for i in range(max_tokens):
        logits = model.get_logits_from_input_ids(ids_list)

        token_id = logits.index(max(logits))

        print(f"Step {i + 1}: token_id={token_id}, decoded={model.decode([token_id])!r}")
        ids_list.append(token_id)

    return model.decode(ids_list)


if __name__ == "__main__":
    model = Small_LLM_Model(device="cpu")
    prompt = "What is the sum of 2 and 3?"
    result = generate(model, prompt, max_tokens=20)
    print("---")
    print("Final result:", repr(result))
