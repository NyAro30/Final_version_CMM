#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   validator.py                                         :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/20 19:47:38 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/26 13:11:14 by mny-aro-           ###   ########.fr      #
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

        ': 0}',
        ': ""}',
        ': true}',
        ': false}',
        ': null}',
        ': {}}',
        ': []}',

        "0",
        "42",
        "-1",
        "true",
        "false",
        "null",
        '""',
        '"x"',
        "}",
        "{}",
        "[]",

        " 0}",

        '": 0}',
        '": ""}',
        '": true}',
        '": false}',
        '": null}',
        '": {}}',
        '": []}',

        ' "x": 0}',
        ' "x": ""}',
        ' "x": true}',

        '""}',
        '{}}',
        '[]}',
    ]

    for suffix in suffixes:
        if is_valid_json(s + suffix):
            return True
    return False


def is_valid_json_object(s: str) -> bool:
    """Return True if s is valid JSON AND parses to a dict."""
    try:
        result = json.loads(s)
        return isinstance(result, dict)
    except json.JSONDecodeError:
        return False


def is_valid_json_object_prefix(s: str) -> bool:
    """Return True if s could be completed to a JSON object (dict)."""
    stripped = s.lstrip()
    if stripped == '':
        return True
    if stripped[0] == '{':
        return is_valid_json_prefix(s)
    return False
