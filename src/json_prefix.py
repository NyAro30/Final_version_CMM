#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   json_prefix.py                                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/26 16:21:30 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/26 16:22:06 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""Prefix validators for JSON scalar values used by constrained decoding."""

import json
import re

NUMBER_PREFIX = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
)


def is_valid_number_prefix(value: str) -> bool:

    """Return whether value is an incomplete or complete JSON number prefix."""

    value = value.lstrip(" \t\r\n")
    return bool(NUMBER_PREFIX.fullmatch(value)) or value in ("", "-")


def is_valid_string_prefix(value: str) -> bool:

    """Return whether value is an incomplete or complete JSON string."""

    value = value.lstrip(" \t\r\n")
    if not value:
        return True
    if value[0] != '"':
        return False
    escaped = False
    unicode_digits = 0
    for index, char in enumerate(value[1:], start=1):
        if unicode_digits:
            if char not in "0123456789abcdefABCDEF":
                return False
            unicode_digits -= 1
        elif escaped:
            if char not in '"\\/bfnrtu':
                return False
            if char == "u":
                unicode_digits = 4
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return not value[index + 1:].strip(" \t\r\n")
        elif ord(char) < 32:
            return False
    return True


def is_valid_boolean_prefix(value: str) -> bool:

    """Return whether value can still become a JSON boolean."""

    value = value.lstrip(" \t\r\n")
    return any(literal.startswith(value) for literal in ("true", "false"))


def scalar_before_separator(value: str, param_type: str) -> str | None:

    """Return a complete typed JSON scalar followed by an object separator."""

    start = len(value) - len(value.lstrip())
    try:
        scalar, end = json.JSONDecoder().raw_decode(value, start)
    except json.JSONDecodeError:
        return None
    expected_type = {
        "number": (int, float),
        "string": (str,),
        "boolean": (bool,),
    }[param_type]
    if not isinstance(scalar, expected_type):
        return None
    if param_type == "number" and isinstance(scalar, bool):
        return None
    remainder = value[end:].lstrip(" \t\r\n")
    if not remainder or remainder[0] not in ",}":
        return None
    if remainder[1:].strip(" \t\r\n"):
        return None
    return value[:end]
