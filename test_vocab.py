#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_vocab.py                                        :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/10 22:16:51 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/14 15:37:42 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json
from llm_sdk.llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load the vocabulary as a dict {token_id: token_string}.

    Args:
        model: The LLM model instance.

    Returns:
        Dict mapping token IDs to their string representations.
    """
    path = model.get_path_to_vocab_file()
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    inverted = {value: key for key, value in data.items()}
    return inverted


if __name__ == "__main__":
    model = Small_LLM_Model(device="cpu")
    vocab = load_vocab(model)

    print("Vocab size:", len(vocab))
    print("First 5 entries:")
    for tid in list(vocab.keys())[:5]:
        print(f"  {tid}: {vocab[tid]!r}")

    print("Token 3555:", repr(vocab.get(3555)))
