#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_json.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/02 15:24:24 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/02 15:53:38 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json

list_dict = [
    {
        "prompt": "first_prompt",
        "name": "name_1",
        "parameters":
        {
            "a": 10
        }
    },
    {
        "prompt": "second_prompt",
        "name": "name_2",
        "parameters":
        {
            "a": 42
        }
    }
]

json_data = json.dumps(list_dict, indent=2)

with open("output.json", "w", encoding="utf-8") as file:
    file.write(json_data)

with open("output.json", "r", encoding="utf-8") as file:
    loaded_data = json.load(file)

print(type(loaded_data))
