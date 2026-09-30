from collections.abc import Iterable, Mapping
from typing import Final

from fastapi import HTTPException

OS_ENVIRON_PREFIX: Final = "os.environ/"


def contains_os_environ_reference(value: object) -> bool:
    stack: Final[list[object]] = [value]  # mutable-ok: local DFS worklist
    seen: Final[set[int]] = set()  # mutable-ok: local cycle guard for the DFS

    while stack:
        src = stack.pop()
        if isinstance(src, str):
            if src.startswith(OS_ENVIRON_PREFIX):
                return True
            continue
        if id(src) in seen:
            continue
        seen.add(id(src))
        if isinstance(src, Mapping):
            values: Iterable[object] = src.values()
        elif isinstance(src, (list, tuple, set, frozenset)):
            values = src
        else:
            continue
        stack.extend(values)
    return False


def reject_os_environ_references(params: dict) -> None:
    """
    Validate that the provided params do not contain any ``os.environ/``
    references. Values with that prefix are expected to come only from
    server-side configuration (already resolved before reaching here). If a
    request-supplied value still carries the prefix, raise ``HTTPException``.
    """
    if not isinstance(params, dict):
        return

    if contains_os_environ_reference(params):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Environment variable references are not permitted in request parameters."
            },  # mutable-ok: HTTPException detail payload
        )
