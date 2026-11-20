#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   vocab.py                                             :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/20 19:48:49 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/20 19:49:02 by mny-aro-           ###   ########.fr      #
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
