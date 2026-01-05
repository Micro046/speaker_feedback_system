# agents/tools/recommendation_tool.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _extract_workflow_result(text: str) -> str:
    cleaned = _strip_ansi(text)
    marker = "Workflow Result:"
    if marker not in cleaned:
        return cleaned.strip()
    tail = cleaned.split(marker, 1)[1]
    lines = [line.rstrip() for line in tail.splitlines()]
    result_lines = []
    for line in lines:
        if line.strip().startswith("-" * 5):
            break
        if line.strip() == "":
            if result_lines:
                result_lines.append("")
            continue
        result_lines.append(line)
    return "\n".join(result_lines).strip()

def _ensure_venv_bin_on_path() -> None:
    venv_bin = str(Path(sys.executable).resolve().parent)
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    if venv_bin not in parts:
        os.environ["PATH"] = os.pathsep.join([venv_bin, path]) if path else venv_bin
        



def _nat_run_help(nat_bin: str) -> str:
    proc = subprocess.run(
        [nat_bin, "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def run_nemo_react_recommendations(
    *,
    config_file: str,
    user_input: str,
    nat_bin: str = "nat",
    cwd: Optional[str] = None,
    log_level: str = "INFO",  # kept for API compatibility, may be unused
) -> str:
    _ensure_venv_bin_on_path()

    if shutil.which(nat_bin) is None:
        raise FileNotFoundError(
            f"'{nat_bin}' not found on PATH. PATH={os.environ.get('PATH','')}"
        )

    cfg = str(Path(config_file).expanduser().resolve())

    help_text = _nat_run_help(nat_bin)

    # Pick the config flag
    if "--config_file" in help_text:
        config_flag = "--config_file"
    elif "--config-file" in help_text:
        config_flag = "--config-file"
    else:
        raise RuntimeError(f"Can't find a config option in `nat run --help`:\n{help_text}")

    # Pick the input flag (prefer file input for long prompts)
    input_file_flag = None
    if "--input_file" in help_text:
        input_file_flag = "--input_file"
    elif "--input-file" in help_text:
        input_file_flag = "--input-file"

    if input_file_flag is None:
        if "--input" in help_text:
            input_flag = "--input"
        elif "--user_input" in help_text:
            input_flag = "--user_input"
        elif "--user-input" in help_text:
            input_flag = "--user-input"
        else:
            raise RuntimeError(f"Can't find an input option in `nat run --help`:\n{help_text}")

    temp_input_path = None
    if input_file_flag is not None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as tmp:
            tmp.write(user_input)
            temp_input_path = tmp.name
        cmd = [nat_bin, "run", config_flag, cfg, input_file_flag, temp_input_path]
    else:
        cmd = [nat_bin, "run", config_flag, cfg, input_flag, user_input]

    # Add logging only if supported (and use the supported spelling)
    if "--log_level" in help_text:
        cmd += ["--log_level", log_level]
    elif "--log-level" in help_text:
        cmd += ["--log-level", log_level]
    # else: don't pass any log option

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if temp_input_path:
            try:
                Path(temp_input_path).unlink(missing_ok=True)
            except OSError:
                pass

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        out = (proc.stdout or "").strip()
        msg = "\n".join(
            [
                f"nat run failed (code={proc.returncode})",
                f"cmd: {cmd}",
                "--- stderr ---",
                err if err else "(empty)",
                "--- stdout ---",
                out if out else "(empty)",
            ]
        )
        raise RuntimeError(msg)

    full_output = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    return _extract_workflow_result(full_output)
