import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from app.core.logger import init_logger  # noqa: E402

init_logger()

if __name__ == "__main__":
    uvicorn.run(
        "app.query.api.query_service:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
    )
