#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   vocab.py                                             :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/20 19:48:49 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/26 09:51:16 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from __future__ import annotations

import json
import os

from .models import VocabularyModel

VERBOSE = os.environ.get("VERBOSE", "") == "1"


def load_vocab(model: VocabularyModel) -> dict[int, str]:

    """Load vocab with DECODED strings (handles special chars correctly).

    Uses model.decode() for each token to get the actual string,
    not the raw vocab.json representation.
    """

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    decoded: dict[int, str] = {}
    total = len(raw_vocab)
    for i, (token_str, token_id) in enumerate(raw_vocab.items()):
        decoded[token_id] = model.decode([token_id])
        if VERBOSE and i % 10000 == 0:
            print(f"Decoding vocab: {i}/{total}")

    return decoded
