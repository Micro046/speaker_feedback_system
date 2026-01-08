# speaker_feedback/nemo/register.py
from __future__ import annotations

"""
This module is intentionally minimal.

You only need a full NAT plugin registration when you want custom NAT function types like:
  _type: speaker_feedback_metric_pack
  _type: speaker_feedback_slide_evidence
etc.

Right now, recommendations.yml uses built-in NAT components:
- react_agent
- chat_completion
So no plugin registration is required.

If you later want custom tools inside NAT, you would:
1) Add entry_points in pyproject.toml (nat.plugins or similar)
2) Implement tool wrappers in speaker_feedback/nemo/tools_nat.py
3) Reference those tool types in YAML (their registered _type tags)
"""

def noop() -> None:
    return
