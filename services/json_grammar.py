from __future__ import annotations

from typing import Any, Dict, List


class GrammarBuilder:
    def __init__(self) -> None:
        self.rules: Dict[str, str] = {}
        self._counter = 0
        self._init_base_rules()

    def _init_base_rules(self) -> None:
        self.rules["root"] = "jsonValue"
        self.rules["ws"] = r'[ \t\n\r]*'
        self.rules["jsonValue"] = (
            'jsonObject | jsonArray | jsonString | jsonNumber | "true" | "false" | "null"'
        )
        self.rules["jsonObject"] = '"{" ws jsonMembers? ws "}"'
        self.rules["jsonMembers"] = "jsonMember (\",\" ws jsonMember)*"
        self.rules["jsonMember"] = "jsonString \":\" ws jsonValue"
        self.rules["jsonArray"] = '"[" ws jsonElements? ws "]"'
        self.rules["jsonElements"] = "jsonValue (\",\" ws jsonValue)*"
        self.rules["jsonString"] = '"\\"" jsonChars "\\""'
        self.rules["jsonChars"] = "jsonChar*"
        self.rules["jsonChar"] = (
            r'[^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)'
        )
        self.rules["hex"] = r'[0-9a-fA-F]'
        self.rules["jsonInteger"] = '"-"? ("0" | [1-9] [0-9]*)'
        self.rules["jsonFrac"] = '"." [0-9]+'
        self.rules["jsonExp"] = '[eE] [+-]? [0-9]+'
        self.rules["jsonNumber"] = "jsonInteger jsonFrac? jsonExp?"

    def build(self, schema: Dict[str, Any]) -> str:
        root_rule = self._schema_to_rule(schema)
        self.rules["root"] = root_rule
        return self._render()

    def _alpha_suffix(self, n: int) -> str:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        s = ""
        while n > 0:
            n, rem = divmod(n - 1, 26)
            s = alphabet[rem] + s
        return s or "a"

    def _safe_prefix(self, prefix: str) -> str:
        cleaned = "".join(ch for ch in prefix if ch.isalpha())
        return cleaned or "rule"

    def _next_name(self, prefix: str) -> str:
        self._counter += 1
        safe_prefix = self._safe_prefix(prefix)
        suffix = self._alpha_suffix(self._counter)
        return f"{safe_prefix}{suffix}"

    def _schema_to_rule(self, schema: Dict[str, Any]) -> str:
        schema_type = schema.get("type")

        if isinstance(schema_type, list):
            return self._rule_for_union(schema, schema_type)

        if schema_type == "object":
            return self._rule_for_object(schema)
        if schema_type == "array":
            return self._rule_for_array(schema)
        if schema_type == "string":
            return "jsonString"
        if schema_type == "number":
            return "jsonNumber"
        if schema_type == "integer":
            return "jsonInteger"
        if schema_type == "boolean":
            return '"true" | "false"'
        if schema_type == "null":
            return '"null"'

        # Fallback to any JSON value
        return "jsonValue"

    def _rule_for_union(self, schema: Dict[str, Any], types: List[str]) -> str:
        name = self._next_name("union")
        rule_names = []
        for t in types:
            rule_names.append(self._schema_to_rule({**schema, "type": t}))
        self.rules[name] = " | ".join(dict.fromkeys(rule_names))
        return name

    def _rule_for_array(self, schema: Dict[str, Any]) -> str:
        name = self._next_name("array")
        items_schema = schema.get("items", {})
        item_rule = self._schema_to_rule(items_schema) if items_schema else "jsonValue"
        items_rule = f"{name}Items"
        self.rules[name] = f'"[" ws {items_rule}? ws "]"'
        self.rules[items_rule] = f"{item_rule} (\",\" ws {item_rule})*"
        return name

    def _rule_for_object(self, schema: Dict[str, Any]) -> str:
        name = self._next_name("object")
        properties: Dict[str, Any] = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        additional_allowed = schema.get("additionalProperties", True)

        prop_names = list(properties.keys())
        required_ordered = [p for p in prop_names if p in required]
        optional_ordered = [p for p in prop_names if p not in required]

        member_rules = {}
        for prop_name, prop_schema in properties.items():
            member_rule = self._next_name(f"member{prop_name}")
            value_rule = self._schema_to_rule(prop_schema)
            key_literal = f'"\\\"{prop_name}\\\""'
            self.rules[member_rule] = f"{key_literal} \":\" ws {value_rule}"
            member_rules[prop_name] = member_rule

        sequences = self._build_sequences(required_ordered, optional_ordered)

        members_rule = f"{name}Members"
        if sequences:
            rendered_sequences = []
            for seq in sequences:
                if not seq:
                    if additional_allowed:
                        rendered_sequences.append("jsonMember (\",\" ws jsonMember)*")
                    continue
                parts = [member_rules[key] for key in seq]
                seq_rule = ' "," ws '.join(parts)
                if additional_allowed:
                    seq_rule = f"{seq_rule} (\",\" ws jsonMember)*"
                rendered_sequences.append(seq_rule)

            if rendered_sequences:
                self.rules[members_rule] = " | ".join(rendered_sequences)
                self.rules[name] = f'"{{" ws {members_rule}? ws "}}"'
            else:
                self.rules[name] = '"{" ws "}"'
        else:
            if additional_allowed:
                self.rules[members_rule] = "jsonMember (\",\" ws jsonMember)*"
                self.rules[name] = f'"{{" ws {members_rule}? ws "}}"'
            else:
                self.rules[name] = '"{" ws "}"'
        return name

    def _build_sequences(self, required: List[str], optional: List[str]) -> List[List[str]]:
        sequences: List[List[str]] = []

        def walk(index: int, current: List[str]) -> None:
            if index == len(optional):
                sequences.append(list(current))
                return
            # skip optional
            walk(index + 1, current)
            # include optional
            current.append(optional[index])
            walk(index + 1, current)
            current.pop()

        base = list(required)
        if optional:
            walk(0, base)
        else:
            sequences.append(base)
        return sequences

    def _render(self) -> str:
        lines = []
        for name, rule in self.rules.items():
            lines.append(f"{name} ::= {rule}")
        return "\n".join(lines)


def schema_to_gbnf(schema: Dict[str, Any]) -> str:
    builder = GrammarBuilder()
    return builder.build(schema)
