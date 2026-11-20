#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   test_parens.py                                       :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/02 19:25:45 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/06 20:55:27 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

def is_balanceable(partial: str) -> bool:
    balanced = 0
    for c in partial:
        if c == "(":
            balanced += 1
        elif c == ")":
            balanced -= 1
        else:
            return False
        if balanced < 0:
            return False
    return True


def valid_next_chars(partial: str) -> list[str]:
    res: list[str] = []
    for c in ["(", ")"]:
        candidate = partial + c

        if is_balanceable(candidate):
            res.append(c)

    return res


if __name__ == "__main__":
    print(is_balanceable(")("))
    print(is_balanceable("))(("))
    print(is_balanceable(")"))
    print(is_balanceable("()"))
    print(is_balanceable("(())"))
    print(is_balanceable(""))
    print(valid_next_chars(")"))
    print(valid_next_chars("("))
