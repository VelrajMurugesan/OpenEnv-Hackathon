"""Server entry point for the GST Invoice Compliance Checker OpenEnv environment."""

import uvicorn

from app.main import app


def main() -> None:
    """Run the OpenEnv environment server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
