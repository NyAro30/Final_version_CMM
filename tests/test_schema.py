import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.json_prefix import (
    is_valid_boolean_prefix,
    is_valid_number_prefix,
    is_valid_string_prefix,
)
from src.models import ParameterDef
from src.schema import (
    generate_parameter_value,
    load_functions,
)


class _EncodedPrompt:
    def tolist(self) -> list[list[int]]:
        return [[]]


class _FakeModel:
    def __init__(
        self, tokens: dict[int, str], sequence: list[int]
    ) -> None:
        self.tokens = tokens
        self.sequence = sequence
        self.decoded_ids: list[int] = []

    def encode(self, text: str) -> _EncodedPrompt:
        return _EncodedPrompt()

    def decode(self, ids: list[int]) -> str:
        self.decoded_ids.extend(ids)
        return "".join(self.tokens[token_id] for token_id in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [-10.0] * len(self.tokens)
        position = len(input_ids)
        if position < len(self.sequence):
            logits[self.sequence[position]] = 10.0
        return logits


class SchemaValidationTests(unittest.TestCase):
    def test_json_prefix_validators(self) -> None:
        valid_numbers = ("", "-", "0", "12.3e-4", "1.", " 2", "\t-3")
        invalid_numbers = ("+1", ".5", "01", "1+2", "1-2")
        for value in valid_numbers:
            with self.subTest(value=value):
                self.assertTrue(is_valid_number_prefix(value))
        for value in invalid_numbers:
            with self.subTest(value=value):
                self.assertFalse(is_valid_number_prefix(value))

        self.assertTrue(is_valid_string_prefix('  "value\\u12'))
        self.assertFalse(is_valid_string_prefix('"value\\u1g'))
        self.assertTrue(is_valid_boolean_prefix(" \ntru"))
        self.assertFalse(is_valid_boolean_prefix("trux"))

    def test_parameter_definitions_reject_unknown_types(self) -> None:
        with self.assertRaises(ValidationError):
            ParameterDef.model_validate({"type": "object"})

    def test_load_functions_skips_only_invalid_definitions(self) -> None:
        definitions = [
            {
                "name": "valid",
                "description": "A valid function.",
                "parameters": {"value": {"type": "string"}},
            },
            {
                "name": "invalid",
                "description": "An invalid function.",
                "parameters": {"value": {"type": "array"}},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "functions.json"
            path.write_text(json.dumps(definitions), encoding="utf-8")
            loaded = load_functions(str(path))
        self.assertEqual(list(loaded), ["valid"])

    def test_generated_values_are_valid_json_primitives(self) -> None:
        cases = (
            ("string", {0: '"', 1: "o", 2: "k"}, [0, 1, 2, 0], '"ok"'),
            ("boolean", {0: "t", 1: "r", 2: "u", 3: "e"},
             [0, 1, 2, 3], "true"),
            ("number", {0: "4", 1: "2", 2: '"'}, [0, 1, 2], "42"),
        )
        for param_type, tokens, sequence, expected in cases:
            with self.subTest(param_type=param_type):
                result = generate_parameter_value(
                    _FakeModel(tokens, sequence),
                    tokens,
                    "prompt",
                    param_type,
                )
                self.assertEqual(result, expected)

    def test_parameter_decoding_stops_at_first_valid_token(self) -> None:
        model = _FakeModel({0: "4", 1: "5", 2: '"'}, [0])
        result = generate_parameter_value(
            model, model.tokens, "prompt", "number", max_tokens=1
        )
        self.assertEqual(result, "4")
        self.assertNotIn(1, model.decoded_ids)

    def test_string_value_can_end_with_a_json_separator_token(self) -> None:
        model = _FakeModel(
            {0: '"', 1: "ok", 2: '",', 3: "later"},
            [0, 1, 2],
        )
        result = generate_parameter_value(
            model, model.tokens, "prompt", "string"
        )
        self.assertEqual(result, '"ok"')


if __name__ == "__main__":
    unittest.main()
