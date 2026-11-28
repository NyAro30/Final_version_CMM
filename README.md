*This project has been created as part of the 42 curriculum by mny-aro-.*

# Call Me Maybe

## Description

Call Me Maybe is an introduction to function calling with large language
models (LLMs). Given a natural-language request and a list of function
definitions, the program uses Qwen3-0.6B to select a function and produce
typed input arguments.

The project focuses on constrained decoding: instead of asking the model for
arbitrary text and trying to repair it afterward, candidate tokens are
filtered while they are generated. Function names must match a defined
function, and argument values must remain valid JSON values of the requested
type.

The program returns a JSON array. Each successful entry has exactly these
fields:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The program builds function calls; it does not execute the selected function
or calculate its return value.

## Instructions

### Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Network access on the first run to download the Qwen model and tokenizer

The supplied `llm_sdk` loads Qwen through PyTorch and Transformers. These
packages are included as runtime dependencies so the program can run with the
provided SDK. This follows the executable SDK requirement
Confirm with the project evaluator which instruction takes
precedence.

### Installation

From the repository root, synchronize the environment:

```sh
uv sync
```

### Execution

Run the mandatory program with the default input and output paths:

```sh
uv run python -m src
```

The equivalent Make target is:

```sh
make run
```

By default, the program reads:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

It creates `data/output/` when needed and writes
`data/output/function_calling_results.json`.

The paths can be overridden:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output /tmp/function_calling_results.json
```

Set `VERBOSE=1` to print the generated function-name and argument tokens:

```sh
VERBOSE=1 uv run python -m src
```

Input files are expected to contain JSON arrays. Invalid input files, invalid
function definitions, model-loading errors, and per-prompt generation errors
are reported to the terminal. Successful prompt results are still written if
another prompt fails; in that case the program exits with a non-zero status.

## Algorithm: constrained decoding

### Function selection

1. The function definitions are loaded from JSON and validated with Pydantic.
   Invalid definitions are reported and skipped.
2. The model receives the available function names and descriptions together
   with the user's request.
3. At each generation step, the model's next-token logits are ranked from
   highest to lowest.
4. Candidate tokens are decoded and retained only if the resulting text is a
   prefix of at least one defined function name. The first valid candidate in
   logit order is selected.
5. Generation stops when the decoded name exactly matches a defined function.

This prevents the function-name generation step from returning an undefined
name.

### Argument generation

1. A prompt describes the selected function, its purpose, its arguments and
   their types, and the user's request. It also explains argument roles such
   as source text, regex pattern, or replacement string.
2. Arguments are generated in definition order. The model is given the
   progressively constructed JSON object prefix, including previously
   generated arguments.
3. At each token step, candidate text is accepted only when it can still form
   a valid JSON scalar of the required type: number, string, or boolean.
   Integer definitions are handled as numeric arguments and converted to
   integers before output.
4. A token may also finish a complete scalar and include the following JSON
   comma or closing brace. The decoder recognizes this boundary and stops at
   the scalar rather than consuming the next field.
5. The final scalar is parsed as JSON and checked against its expected Python
   type. Non-finite floating-point numbers and mismatched types are rejected.

The prefix checks are implemented in `src/json_prefix.py`; function and
argument generation are in `src/schema.py`. This is token-level constrained
decoding, not prompt-only formatting.

## Design decisions

- **Use the supplied SDK:** `Small_LLM_Model` provides model inference. The
  implementation calls its public `encode`, `decode`,
  `get_logits_from_input_ids`, and vocabulary-path methods.
- **Keep generation constrained:** prompts guide the model's interpretation,
  while token filtering enforces function-name and JSON-type boundaries.
- **Use Pydantic at input/output boundaries:** function definitions and
  prompts are validated before inference, and generated results are
  validated before they are written.
- **Keep partial successes:** a failing prompt does not discard results from
  other prompts. Errors are reported and the final exit code remains
  non-zero if any prompt failed.
- **Write strict JSON:** `allow_nan=False` prevents non-standard NaN or
  infinity values from appearing in the output.
- **Keep the model on CPU:** this avoids requiring a GPU, at the cost of
  slower inference.
- **Do not execute tools:** the program returns a selected function and its
  arguments; it does not call arbitrary functions.

## Performance analysis

On the provided 11-prompt input file, a recent CPU run completed in about
86 seconds and produced 11 parseable results. The function names, argument
keys, and argument types matched the definitions. The three generated regex
calls were also applied to their source strings and produced the intended
substitutions.

This is a small example set, not evidence of greater than 90% accuracy on
unseen or ambiguous prompts. The model can still misunderstand a request and
produce a valid value of the wrong meaning. CPU speed also depends on the
machine and cache state; the first run includes model download and loading.
The generation is greedy under token constraints, but this alone does not
guarantee semantic correctness on every run or input.

## Challenges faced

- **Syntax versus meaning:** constrained decoding can ensure a number or
  string is valid JSON, but it cannot by itself ensure that the value answers
  the user's request. Function and argument descriptions therefore provide
  semantic context.
- **JSON token boundaries:** a tokenizer may emit a string-closing quote and
  the next object's comma or brace in the same token. The decoder must detect
  that boundary without rejecting the token or accidentally including the
  separator in the argument.
- **Model performance:** evaluating every token in a 150,000-token vocabulary
  is expensive. Candidates are visited in descending logit order, and
  selection stops at the first valid token.
- **Conflicting dependency instructions:** the supplied SDK requires
  PyTorch, Transformers, and Hugging Face packages, while the subject also
  prohibits those dependencies. They are currently declared so the mandatory
  program runs; this exception needs confirmation from the evaluator.

## Testing strategy

Validation has included:

- Running `uv sync` and `uv lock --check`.
- Running `make run` on the provided input file.
- Parsing the resulting JSON and checking all 11 entries, their exact output
  fields, selected function names, required argument keys, and argument
  types.
- Applying each generated regex with Python's `re.sub` to verify the three
  substitution examples; in the measured run, each produced the intended
  transformed text.
- `make test` is declared in the Makefile, but the current repository
  snapshot has no tracked `tests/` directory, so that target cannot discover
  tests.
- Static checks are available through `make lint` and `make lint-strict`.
  The current local `src/schema.py` has outstanding formatting findings that
  should be corrected before evaluation.

The generated values should still be reviewed against the intended meaning:
JSON parseability and type validity are necessary, but not sufficient,
correctness checks.

Function parameters currently support primitive types. The input schema does
not define enum or allowed-value constraints, so such constraints are not
currently enforced by this implementation.

## Example usage

For the request `What is the sum of 2 and 3?`, the program selects
`fn_add_numbers` and returns:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

To run with a custom output path:

```sh
uv run python -m src --output /tmp/calls.json
```

## Resources

- [42 School](https://www.42.fr/) — curriculum and peer-evaluation context.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model
  information.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
  — causal language model concepts and inference.
- [Python `json` documentation](https://docs.python.org/3/library/json.html)
  — JSON parsing and serialization.
- [Python `re` documentation](https://docs.python.org/3/library/re.html) —
  regular-expression syntax and substitution.
- [Pydantic documentation](https://docs.pydantic.dev/) — data validation.
- [uv documentation](https://docs.astral.sh/uv/) — Python project and
  dependency management.

### Use of AI

AI assistance (GitHub Copilot) was used during development to help diagnose
Python import and Makefile issues, review implementation behavior, suggest
constrained-decoding and prompt improvements, and draft documentation. It was
also used to help design validation scenarios. Generated suggestions and
code were reviewed and tested by the student; AI was not used as a substitute
for checking the program's outputs against the provided examples.
