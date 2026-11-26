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
