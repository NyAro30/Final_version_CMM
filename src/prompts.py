#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   prompts.py                                           :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/26 16:22:30 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/26 16:22:32 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

"""Prompt construction for function and argument selection."""


def chat_prompt(instruction: str, request: str) -> str:
    """Format a prompt for the supplied Qwen chat model."""
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n</think>\n"
    )


def arguments_prompt(
    function_name: str,
    description: str,
    user_request: str,
    parameters: dict[str, str],
) -> str:

    """Describe a function's complete input object and argument roles."""

    parameter_names = list(parameters)
    prompt = (
        f"Function {function_name}: {description}\n"
        "Arguments in order: "
        + ", ".join(
            f"{name} ({parameter_type})"
            for name, parameter_type in parameters.items()
        )
        + "\n"
        f"Request: {user_request}\n"
        "Return exactly these arguments as a JSON object, not the "
        "transformed result. Every value is a function input. For neutral "
        "argument names, map request values to arguments in order. Do not "
        "invent, normalize, or transform inputs."
    )
    roles: list[str] = []
    for name in parameter_names:
        lower_name = name.lower()
        if "regex" in lower_name or "pattern" in lower_name:
            roles.append(
                f"{name} is a Python regex matching the feature requested; "
                "write it without slash delimiters. It is a reusable "
                "pattern, not the actual matched text or feature name. "
                "When the feature is one character from a set, match every "
                "member of that set."
            )
        elif "replacement" in lower_name:
            roles.append(
                f"{name} is the literal inserted at each match; use the "
                "replacement specified by the request, not the match. "
                "When a symbol is named, return its glyph once rather than "
                "its English name. A symbol is one literal character."
            )
        elif "source" in lower_name:
            roles.append(
                f"{name} is the exact original text; copy it unchanged."
            )
        elif "name" in lower_name:
            roles.append(
                f"{name} is the person or entity named by the request; "
                "preserve its spelling and capitalization."
            )
    if roles:
        prompt += " Argument roles: " + " ".join(roles)
    if len(parameter_names) > 1:
        prompt += (
            " For neutral names, map request values to arguments in the "
            "listed order. Do not invent or transform inputs."
        )
    if any(param_type == "number" for param_type in parameters.values()):
        prompt += (
            " Generic example: for add(left, right) and 'add 7 and 9', "
            "left is 7 and right is 9. Return input numbers, not results."
        )
    if any(param_type == "string" for param_type in parameters.values()):
        if not roles:
            prompt += (
                " Generic example: for reverse(text) and 'reverse moss', "
                "text is \"moss\"."
            )
    return prompt
