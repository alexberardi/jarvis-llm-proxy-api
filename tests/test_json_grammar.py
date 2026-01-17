import re

from services.json_grammar import schema_to_gbnf


def _sample_schema():
    return {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
            "error": {
                "type": ["object", "null"],
                "properties": {
                    "type": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["type", "message"],
            },
        },
        "required": ["message", "tool_calls"],
        "additionalProperties": True,
    }


def test_schema_to_gbnf_rule_names_are_letters_only():
    grammar = schema_to_gbnf(_sample_schema())
    lines = [line for line in grammar.splitlines() if line.strip()]
    assert lines, "Expected non-empty grammar output"

    rule_name_re = re.compile(r"^[A-Za-z]+$")
    seen = set()
    for line in lines:
        assert "::=" in line, f"Expected '::=' in rule line: {line}"
        rule_name = line.split("::=", 1)[0].strip()
        assert rule_name_re.match(rule_name), f"Invalid rule name: {rule_name}"
        assert rule_name not in seen, f"Duplicate rule name: {rule_name}"
        seen.add(rule_name)
