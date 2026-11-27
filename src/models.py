#!/usr/bin/env python3
# ########################################################################### #
#   shebang: 1                                                                #
#                                                          :::      ::::::::  #
#   models.py                                            :+:      :+:    :+:  #
#                                                      +:+ +:+         +:+    #
#   By: mny-aro- <mny-aro-@student.42antananarivo.   +#+  +:+       +#+       #
#                                                  +#+#+#+#+#+   +#+          #
#   Created: 2026/11/22 20:05:17 by mny-aro-            #+#    #+#            #
#   Updated: 2026/11/28 00:54:02 by mny-aro-           ###   ########.fr      #
#                                                                             #
# ########################################################################### #

from typing import Any, Literal, Protocol
from pydantic import BaseModel, Field


class EncodedInput(Protocol):

    """Minimal tokenizer output interface used by the decoder."""

    def tolist(self) -> list[list[int]]:

        """Return the encoded token IDs as nested Python lists."""


class LLMModel(Protocol):

    """Public model operations required by constrained decoding."""

    def encode(self, text: str) -> EncodedInput:

        """Encode text into model input IDs."""

    def decode(self, ids: list[int]) -> str:

        """Decode token IDs into text."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:

        """Return next-token logits for the supplied token IDs."""


class VocabularyModel(LLMModel, Protocol):

    """Model operations required to load its vocabulary."""

    def get_path_to_vocab_file(self) -> str:

        """Return the path to the model's vocabulary file."""


class ParameterDef(BaseModel):

    """Definition of a single function parameter."""

    type: Literal["number", "string", "boolean", "integer"] = Field(
        description="Parameter type"
    )


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
