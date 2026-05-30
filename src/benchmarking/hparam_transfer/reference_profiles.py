"""Reference profile export helpers for DPG-HPT."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from src.benchmarking.config.loader import _load_structured_file

SUPPORTED_STRUCTURED_SUFFIXES = {".json", ".yaml", ".yml"}
SUPPORTED_TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml"}


def _field_variants(field_path: str) -> list[str]:
    parts = field_path.split(".")
    variants = [field_path]
    for index in range(1, len(parts)):
        variants.append(".".join(parts[index:]))
    return variants


def _get_nested_value(payload: Any, field_path: str) -> Any:
    current = payload
    for token in field_path.split("."):
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(field_path)
            current = current[token]
        else:
            raise KeyError(field_path)
    return current


def _resolve_line_number(text: str, field_path: str) -> int | None:
    tail = field_path.split(".")[-1]
    for index, line in enumerate(text.splitlines(), start=1):
        if field_path in line:
            return index
    key_pattern = re.compile(
        rf"(^|[^A-Za-z0-9_])['\"]?{re.escape(tail)}['\"]?\s*[:=]"
    )
    for index, line in enumerate(text.splitlines(), start=1):
        if key_pattern.search(line):
            return index
    return None


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_scalar(item) for key, item in value.items()}
    return str(value)


def _normalize_route_field_spec(field_spec: Any) -> dict[str, Any]:
    if isinstance(field_spec, str):
        return {
            "parameter": field_spec,
            "reference_field": field_spec,
            "note": None,
        }
    if isinstance(field_spec, dict):
        parameter = field_spec.get("parameter") or field_spec.get("field")
        reference_field = field_spec.get("reference_field") or parameter
        if not parameter or not reference_field:
            raise ValueError("Route field spec requires parameter and reference_field.")
        return {
            "parameter": str(parameter),
            "reference_field": str(reference_field),
            "note": field_spec.get("note"),
        }
    raise TypeError("Route field spec must be a string or mapping.")


def _safe_eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute):
        chain = _attribute_chain(node)
        if chain is not None and chain in env:
            return env[chain]
    if isinstance(node, ast.Dict):
        keys = [_safe_eval(key, env) for key in node.keys]
        values = [_safe_eval(value, env) for value in node.values]
        return dict(zip(keys, values))
    if isinstance(node, ast.List):
        return [_safe_eval(element, env) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(element, env) for element in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _safe_eval(node.operand, env)
        if isinstance(operand, (int, float)):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, env)
        right = _safe_eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"dict", "edict", "EasyDict"}
        ):
            if node.args:
                first_arg = _safe_eval(node.args[0], env)
                if isinstance(first_arg, dict):
                    base = dict(first_arg)
                else:
                    raise ValueError("Unsupported positional call argument.")
            else:
                base = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                base[keyword.arg] = _safe_eval(keyword.value, env)
            return base
    raise ValueError(
        "Unsupported AST node for static extraction: "
        f"{type(node).__name__}"
    )


def _attribute_chain(node: ast.AST) -> str | None:
    tokens: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        tokens.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        tokens.append(current.id)
        return ".".join(reversed(tokens))
    return None


def _relative_import_paths(path: Path, tree: ast.Module) -> list[Path]:
    paths: list[Path] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level <= 0 or not node.module:
            continue
        candidate = path.parent.joinpath(*node.module.split(".")).with_suffix(".py")
        if candidate.exists():
            paths.append(candidate.resolve())
    return paths


def _collect_python_assignments(
    path: Path,
    seen: set[Path] | None = None,
) -> dict[str, tuple[Any, Path]]:
    path = path.resolve()
    if seen is None:
        seen = set()
    if path in seen:
        return {}
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    env: dict[str, Any] = {}
    assignments: dict[str, tuple[Any, Path]] = {}

    for import_path in _relative_import_paths(path, tree):
        imported = _collect_python_assignments(import_path, seen)
        assignments.update(imported)
        for key, (value, _) in imported.items():
            env[key] = value

    def visit_body(body: list[ast.stmt], prefix: str = "") -> None:
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                try:
                    value = _safe_eval(stmt.value, env)
                except ValueError:
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
                        assignments[prefix + target.id] = (value, path)
                    else:
                        chain = _attribute_chain(target)
                        if chain is not None:
                            env[chain] = value
                            env[prefix + chain] = value
                            assignments[prefix + chain] = (value, path)
            elif isinstance(stmt, ast.ClassDef):
                visit_body(stmt.body, prefix=f"{prefix}{stmt.name}.")
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_body(stmt.body, prefix=f"{prefix}{stmt.name}.")

    visit_body(tree.body)
    return assignments


def _extract_from_python(path: Path, field_path: str) -> tuple[Any, Path]:
    assignments = _collect_python_assignments(path)
    for variant in _field_variants(field_path):
        if variant in assignments:
            return assignments[variant]
    for key, (value, source_path) in assignments.items():
        if key.endswith(f".{field_path}") or key.endswith(
            f".{field_path.split('.')[-1]}"
        ):
            return value, source_path
        if isinstance(value, dict):
            tail = key.split(".")[-1]
            prefix = f"{tail}."
            if field_path.startswith(prefix):
                nested_field = field_path[len(prefix) :]
                try:
                    return _get_nested_value(value, nested_field), source_path
                except KeyError:
                    continue
            for variant in _field_variants(field_path):
                try:
                    return _get_nested_value(value, variant), source_path
                except KeyError:
                    continue
    raise KeyError(field_path)


def extract_reference_field(config_path: str | Path, field_path: str) -> dict[str, Any]:
    """Extract one field from a route config and attach source evidence."""

    path = Path(config_path).resolve()
    text = path.read_text(encoding="utf-8")
    result: dict[str, Any] = {
        "field": field_path,
        "config_path": str(path),
        "line": _resolve_line_number(text, field_path),
        "status": "ok",
        "value": None,
    }

    try:
        source_path = path
        if path.suffix.lower() in SUPPORTED_STRUCTURED_SUFFIXES:
            payload = _load_structured_file(path)
            value = None
            for variant in _field_variants(field_path):
                try:
                    value = _get_nested_value(payload, variant)
                    break
                except KeyError:
                    continue
            if value is None:
                raise KeyError(field_path)
        elif path.suffix.lower() == ".py":
            value, source_path = _extract_from_python(path, field_path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        if source_path != path:
            result["config_path"] = str(source_path)
            result["line"] = _resolve_line_number(
                source_path.read_text(encoding="utf-8"),
                field_path,
            )
        result["value"] = _normalize_scalar(value)
    except FileNotFoundError:
        result["status"] = "missing_config"
        result["error"] = f"Config file not found: {path}"
    except KeyError:
        result["status"] = "missing_field"
        result["error"] = f"Field not found: {field_path}"
    except ValueError as exc:
        result["status"] = "unsupported_format"
        result["error"] = str(exc)
    return result


def extract_reference_parameter(
    config_path: str | Path,
    field_spec: Any,
) -> dict[str, Any]:
    """Extract one benchmark parameter and attach vendor evidence metadata."""

    spec = _normalize_route_field_spec(field_spec)
    record = extract_reference_field(config_path, spec["reference_field"])
    record["parameter"] = spec["parameter"]
    record["reference_field"] = spec["reference_field"]
    if spec["note"] is not None:
        record["note"] = spec["note"]
    return record


def export_reference_profiles(
    model_id: str,
    route_registry_path: str | Path,
) -> dict[str, Any]:
    """Export reference profiles for one model using its route card."""

    route_registry = yaml.safe_load(
        Path(route_registry_path).read_text(encoding="utf-8")
    )
    model_routes = route_registry["models"][model_id]
    route_bundle: list[dict[str, Any]] = []
    for route in model_routes.get("known_routes", []):
        config_path = route["config"]
        field_specs = model_routes.get("scale_sensitive_params", [])
        extractions = [
            extract_reference_parameter(config_path=config_path, field_spec=field_spec)
            for field_spec in field_specs
        ]
        route_payload = {
            "dataset": route["dataset"],
            "route": route["route"],
            "config_path": config_path,
            "confidence": route.get("confidence"),
            "fields": extractions,
            "summary": {
                "field_count": len(extractions),
                "resolved_count": sum(
                    1 for item in extractions if item["status"] == "ok"
                ),
                "missing_count": sum(
                    1 for item in extractions if item["status"] != "ok"
                ),
            },
        }
        route_bundle.append(route_payload)

    payload = {
        "schema_version": 1,
        "model": model_id,
        "vendor_path": model_routes.get("vendor_path"),
        "source_kind": model_routes.get("source_kind"),
        "route_count": len(route_bundle),
        "routes": route_bundle,
        "digests": {
            "route_registry_digest": hashlib.sha256(
                Path(route_registry_path).read_bytes()
            ).hexdigest(),
        },
        "known_risks": model_routes.get("known_risks", []),
    }
    payload["digests"]["reference_profile_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def render_reference_profiles_markdown(bundle: dict[str, Any]) -> str:
    """Render a compact markdown summary for exported reference profiles."""

    lines = [
        f"# Reference Profiles: {bundle['model']}",
        "",
        f"- source_kind: `{bundle['source_kind']}`",
        f"- vendor_path: `{bundle['vendor_path']}`",
        f"- route_count: `{bundle['route_count']}`",
        "",
    ]
    for route in bundle["routes"]:
        lines.extend(
            [
                f"## {route['dataset']} ({route['route']})",
                "",
                f"- config_path: `{route['config_path']}`",
                f"- confidence: `{route['confidence']}`",
                f"- resolved_count: `{route['summary']['resolved_count']}`",
                f"- missing_count: `{route['summary']['missing_count']}`",
                "",
                "| Field | Status | Value | Line |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for field in route["fields"]:
            value = json.dumps(field["value"], ensure_ascii=True)
            lines.append(
                "| "
                f"{field['parameter']} <= {field['reference_field']} | "
                f"{field['status']} | {value} | {field['line']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_reference_profile_outputs(
    bundle: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown outputs for one model reference bundle."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{bundle['model']}_reference_profiles.json"
    md_path = output_dir / f"{bundle['model']}_reference_profiles.md"
    json_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_reference_profiles_markdown(bundle),
        encoding="utf-8",
    )
    return json_path, md_path
