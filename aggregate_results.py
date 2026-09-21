"""Compatibility entry point for :mod:`aggregate_job_outputs`.

Use this short, meaningful name from cluster post-processing scripts:
``python aggregate_results.py --root .``.
"""
from aggregate_job_outputs import aggregate, main

__all__ = ["aggregate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
