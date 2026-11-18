from __future__ import annotations

import logging
from typing import Any

from blindnav.utils.device import free_memory

log = logging.getLogger(__name__)


class LazyModel:
    """Loads weights on first use; ``unload()`` frees GPU memory."""

    name = "model"

    def __init__(self) -> None:
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        if self._model is None:
            log.info("Loading %s", self.name)
            self._model = self._build()
        return self._model

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            free_memory()

    def _build(self) -> Any:
        raise NotImplementedError
