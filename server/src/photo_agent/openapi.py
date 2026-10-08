"""Write the OpenAPI schema to disk so the web client can be generated from it.

Usage: uv run python -m photo_agent.openapi [path]
"""

import json
import sys
from pathlib import Path

from photo_agent.main import app
from photo_agent.settings import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "api" / "openapi.json"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(app.openapi(), indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
