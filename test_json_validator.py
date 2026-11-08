#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_json_validator.py                               :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/08 08:23:04 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/08 18:13:23 by mny-aro-           ###   ########.fr      #
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


def is_valid_json_prefix(s: str) -> bool:
    """Return True if s could be completed to valid JSON"""
    if is_valid_json(s):
        return True

    suffixes = [
        "}", "", "0}", "true}", "false}", "null}",
    ]

    for suffix in suffixes:
        if is_valid_json(s + suffix):
            return True
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

    print("=== Test suffix ===")
    prefix_tests = [
        ('', True),
        ('{', True),
        ('{"', True),
        ('{"key"', True),
        ('{"key":', True),
        ('{"key": 1', True),
        ('{"key": 1}', True),
        ('{"key": 1,', True),
        ('{"key": 1, "key2":', True),
        ('}', False),
        ('{key}', False),
        ('{"key": 1,}', False),
        ('{"key":}', False),
        ('}{}', False),
        ('{,', False),
    ]

    for s, expected in prefix_tests:
        result = is_valid_json_prefix(s)
        status = "OK" if result == expected else "FAIL"
        print(f"{status}: is_valid_json_prefix({s!r}) = {result}, expected {expected}")
