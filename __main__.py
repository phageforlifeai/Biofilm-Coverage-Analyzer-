"""CLI entrypoint for `python -m biofilm`."""

from .analyzer import main

if __name__ == "__main__":
    raise SystemExit(main())
