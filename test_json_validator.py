#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_json_validator.py                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/08 08:23:04 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/08 16:44:06 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

import json

def is_valid_json(s: str) -> bool:
    """Return True if s is valid complete JSON, False otherwise."""
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


if __name__ == "__main__":
    tests = [
        ('{"a": 1}', True),
        ('42', True),
        ('"hello"', True),
        ('true', True),
        ('null', True),
        ('[1, 2, 3]', True),
        ('{"a": {"b": 2}}', True),
        ('', False),
        ('{a: 1}', False),
        ('{"a": 1,}', False),
        ('{"a": 1', False),
        ('not json', False),
        ('{"a": 1} extra', False),
    ]

    for s, expected in tests:
        result = is_valid_json(s)
        status = "OK" if result == expected else "FAIL"
        print(f"{status}: is_valid_json({s!r}) = {result}, expected {expected}")
