from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # OpenAI Python SDK (v1+)
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


def _repo_root() -> Path:
    # src/agent/router_llm.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_capabilities_json(path: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path) if path else (_repo_root() / "pipeline_capabilities.json")
    return json.loads(p.read_text(encoding="utf-8"))


def _capability_ids(capabilities_json: Dict[str, Any]) -> List[str]:
    caps = capabilities_json.get("capabilities", [])
    out: List[str] = []
    for c in caps:
        cid = c.get("id")
        if isinstance(cid, str) and cid.strip():
            out.append(cid.strip())
    return out


def llm_choose_capability(
    *,
    request: str,
    csv_columns: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    capabilities_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return ONLY:
      {"capability_id": "...", "reason": "..."}.

    Minimal behavior:
      - Loads capability IDs from pipeline_capabilities.json
      - Asks the LLM to choose ONE id + a short reason
      - Validates the chosen id is in the registry; otherwise falls back to first id
    """
    caps_json = _load_capabilities_json(capabilities_path)
    allowed = _capability_ids(caps_json)
    if not allowed:
        raise RuntimeError("No capabilities found in pipeline_capabilities.json (missing 'id' fields).")

    # Hard fallback (safe default)
    fallback_id = allowed[0]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "capability_id": fallback_id,
            "reason": "OPENAI_API_KEY not set; defaulting to first capability.",
        }

    if OpenAI is None:
        return {
            "capability_id": fallback_id,
            "reason": "openai package not available; defaulting to first capability.",
        }

    cols_text = ""
    if csv_columns:
        cols_text = "\nCSV columns:\n- " + "\n- ".join(csv_columns)

    # Keep prompt extremely simple and constrained.
    system = (
        "You are a strict router. Choose exactly ONE capability_id from the allowed list. "
        "Return valid JSON with keys: capability_id, reason. No extra keys."
    )
    user = (
        f"Allowed capability_id values:\n{allowed}\n\n"
        f"User request:\n{request}\n"
        f"{cols_text}\n\n"
        "Return JSON only."
    )

    client = OpenAI(api_key=api_key)

    # Ask for JSON; keep it robust: try response_format JSON if supported, else parse plain text.
    content = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception:
        # Fallback without response_format
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()

    # Parse
    try:
        obj = json.loads(content)
    except Exception:
        return {
            "capability_id": fallback_id,
            "reason": "LLM returned non-JSON; defaulting to first capability.",
        }

    cap_id = str(obj.get("capability_id", "")).strip()
    reason = str(obj.get("reason", "")).strip() or "No reason provided."

    if cap_id not in allowed:
        return {
            "capability_id": fallback_id,
            "reason": f"LLM chose invalid capability_id='{cap_id}'; defaulting to '{fallback_id}'.",
        }

    return {"capability_id": cap_id, "reason": reason}
