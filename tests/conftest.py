from __future__ import annotations

import sys
from pathlib import Path

# When running via the `pytest` console script, the repository root is not always on
# `sys.path`. Add it so imports like `import beginoi` work reliably.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
