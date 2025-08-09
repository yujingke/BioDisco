# libraries.py
import uuid
from typing import Dict, Any, List, Tuple

class BaseLibrary:
    # Base class for storing records with unique id.
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def _generate_id(self) -> str:
        # Generate a unique id.
        return uuid.uuid4().hex

    def _exists(self, content: Any) -> Tuple[bool, str]:
        # To be overridden by subclasses for deduplication.
        return False, ""

    def add(self, content: Any) -> str:
        # Add content if not duplicate; else return existing id.
        exists, existing_id = self._exists(content)
        if exists:
            return existing_id
        new_id = self._generate_id()
        self._store[new_id] = content
        return new_id

    def get(self, id_: str) -> Any:
        # Get content by id.
        return self._store.get(id_)

    def all(self) -> Dict[str, Any]:
        # Get all records.
        return dict(self._store)



class HypothesisLibrary(BaseLibrary):
    # Store hypotheses, deduplicate by text.
    def __init__(self):
        super().__init__()
        self._text_index: Dict[str, str] = {}

    def _exists(self, content: str) -> Tuple[bool, str]:
        if content in self._text_index:
            return True, self._text_index[content]
        return False, ""

    def add(self, content: str) -> str:
        if not isinstance(content, str):
            raise ValueError("HypothesisLibrary content must be str.")
        exists, existing_id = self._exists(content)
        if exists:
            return existing_id
        new_id = self._generate_id()
        self._store[new_id] = content
        self._text_index[content] = new_id
        return new_id

