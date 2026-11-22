#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   models.py                                            :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/22 20:05:17 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/22 20:39:33 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import Any
from pydantic import BaseModel, Field

class ParameterDef(BaseModel):
    
    """Definition of a single function parameter."""
    
    type: str = Field(description="Parameter type: number, string, or boolean")


class FunctionDef(BaseModel):

    """Definition of a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterDef] = Field(default_factory=dict)
    returns: dict[str, str] = Field(default_factory=dict)


class InputPrompt(BaseModel):

    """A single input prompt from the test file."""

    prompt: str


class OutputResult(BaseModel):

    """A single output function call result."""

    prompt: str
    name: str
    parameters: dict[str, Any]
