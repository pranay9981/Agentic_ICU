from __future__ import annotations

import uvicorn

from agentic_icu.config import settings


def main() -> None:
    uvicorn.run(
        "agentic_icu.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
