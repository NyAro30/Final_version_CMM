#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_valid_tokens.py                                 :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/14 16:25:08 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/20 22:35:41 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from llm_sdk.llm_sdk import Small_LLM_Model
from Test.test_json_validator import is_valid_json_object_prefix
from Test.test_vocab import load_vocab


def valid_next_tokens(prefix: str, vocab: dict[int, str]) -> list[int]:
    """Return token IDs that, when appended to prefix, keep it a valid JSON
    prefix.

    Args:
        prefix: Current JSON prefix generated so far.
        vocab: Dict {token_id: token_string}.

    Returns:
        List of valid token IDs.
    """

    valid = []
    for token_id, token_str in vocab.items():
        clean_str = token_str.replace("Ġ", " ")
        if is_valid_json_object_prefix(prefix + clean_str):
            valid.append(token_id)
    return valid


if __name__ == "__main__":
    model = Small_LLM_Model(device="cpu")
    vocab = load_vocab(model)

    test_prefixes = [
        "",
        "{",
        '{"name"',
        '{"name":',
        '{"name": "fn_add',
    ]

    for prefix in test_prefixes:
        valid = valid_next_tokens(prefix, vocab)
        print(f"\nPrefix: {prefix!r}")
        print(f"  Valid token count: {len(valid)}")
        for tid in valid[:5]:
            print(f"    {tid}: {model.decode([tid])!r}")
