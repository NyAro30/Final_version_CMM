#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_llm.py                                          :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/06 21:50:33 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/18 14:50:26 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from llm_sdk.llm_sdk import Small_LLM_Model

model = Small_LLM_Model(device="cpu")

prompt = "What is the sum of 2 and 3?"
ids_tensor = model.encode(prompt)

ids_list = ids_tensor.tolist()[0]

logits = model.get_logits_from_input_ids(ids_list)

print("Type:", type(logits))
print("Len:", len(logits))
print("First 5:", logits[:5])
print("Index of max:", logits.index(max(logits)))

predicted_token_id = logits.index(max(logits))
print("Predicted next token:", repr(model.decode([predicted_token_id])))
