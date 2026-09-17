"""Deterministic schema enforcement, JSON repair and boundary transformation.

Handles common micro-LLM / compact model structural defects:
- Trailing commas, unquoted keys, markdown backtick fences
- Missing required fields with deterministic defaults
- Type coercion (strings -> numbers/booleans where safe)
- Value range clamping and forbidden field stripping

Zero external dependencies: pure Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set, Type


@dataclass
class SchemaRule:
    expected_type: Type
    required: bool = True
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None


@dataclass
class AdapterReport:
    is_valid: bool
    data: Dict[str, Any]
    repaired_syntax: bool = False
    coerced_fields: List[str] = field(default_factory=list)
    clamped_fields: List[str] = field(default_factory=list)
    dropped_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class JSONTransformer:
    """Deterministic transformer and repair engine for structured JSON payloads."""
    
    def __init__(self, schema: Dict[str, SchemaRule], *, allow_extra_keys: bool = False):
        self.schema = schema
        self.allow_extra_keys = allow_extra_keys
        
    @staticmethod
    def clean_raw_text(text: str) -> str:
        """Strip markdown fences, leading/trailing prose, and common syntax flaws."""
        s = text.strip()
        # Remove markdown fences
        if s.startswith('```'):
            lines = s.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            s = chr(10).join(lines).strip()
            
        # Extract first outer JSON object {...}
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            s = s[start:end + 1]
            
        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s
        
    def transform(self, raw_input: str | Dict[str, Any]) -> AdapterReport:
        repaired_syntax = False
        if isinstance(raw_input, str):
            cleaned = self.clean_raw_text(raw_input)
            try:
                parsed = json.loads(cleaned)
                repaired_syntax = (cleaned != raw_input.strip())
            except Exception as e:
                return AdapterReport(False, {}, repaired_syntax=False, errors=[f"Syntax parse failed: {e}"])
        elif isinstance(raw_input, dict):
            parsed = dict(raw_input)
        else:
            return AdapterReport(False, {}, errors=["Input must be str or dict"])
            
        if not isinstance(parsed, dict):
            return AdapterReport(False, {}, errors=["Parsed root is not a dictionary object"])
            
        output: Dict[str, Any] = {}
        coerced: List[str] = []
        clamped: List[str] = []
        dropped: List[str] = []
        errors: List[str] = []
        
        # Enforce schema
        for key, rule in self.schema.items():
            if key not in parsed:
                if rule.default is not None:
                    output[key] = rule.default
                    coerced.append(f"{key}: assigned default")
                elif rule.required:
                    errors.append(f"Missing required field '{key}'")
                continue
            
            val = parsed[key]
            # Type verification and safe coercion
            if not isinstance(val, rule.expected_type):
                try:
                    if rule.expected_type in (int, float):
                        val = rule.expected_type(val)
                        coerced.append(f"{key}: coerced to {rule.expected_type.__name__}")
                    elif rule.expected_type == bool and isinstance(val, str):
                        val = val.strip().lower() in ('true', '1', 'yes')
                        coerced.append(f"{key}: coerced str to bool")
                    elif rule.expected_type == str:
                        val = str(val)
                        coerced.append(f"{key}: coerced to str")
                    else:
                        errors.append(f"Field '{key}' cannot be converted to {rule.expected_type.__name__}")
                        continue
                except (ValueError, TypeError):
                    errors.append(f"Type conversion failed for field '{key}'")
                    continue
                    
            # Clamping for numeric values
            if isinstance(val, (int, float)):
                if rule.min_value is not None and val < rule.min_value:
                    val = rule.expected_type(rule.min_value)
                    clamped.append(f"{key}: clamped to min {rule.min_value}")
                elif rule.max_value is not None and val > rule.max_value:
                    val = rule.expected_type(rule.max_value)
                    clamped.append(f"{key}: clamped to max {rule.max_value}")
                    
            # Allowed values set
            if rule.allowed_values is not None and val not in rule.allowed_values:
                errors.append(f"Value '{val}' for '{key}' not in allowed set {rule.allowed_values}")
                continue
                
            output[key] = val
            
        # Extra keys handling
        for k in list(parsed.keys()):
            if k not in self.schema:
                if self.allow_extra_keys:
                    output[k] = parsed[k]
                else:
                    dropped.append(k)
                    
        is_valid = len(errors) == 0
        return AdapterReport(
            is_valid=is_valid,
            data=output,
            repaired_syntax=repaired_syntax,
            coerced_fields=coerced,
            clamped_fields=clamped,
            dropped_fields=dropped,
            errors=errors,
        )


def repair_and_validate_json(raw_input: str | Dict[str, Any], schema: Dict[str, SchemaRule], **kwargs) -> AdapterReport:
    transformer = JSONTransformer(schema, **kwargs)
    return transformer.transform(raw_input)
