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

class BackgroundLibrary(BaseLibrary):
    # Store background text, deduplicate by full text.
    def __init__(self):
        super().__init__()
        self._reverse_index: Dict[str, str] = {}

    def _exists(self, content: str) -> Tuple[bool, str]:
        if content in self._reverse_index:
            return True, self._reverse_index[content]
        return False, ""

    def add(self, content: str) -> str:
        exists, existing_id = self._exists(content)
        if exists:
            return existing_id
        new_id = self._generate_id()
        self._store[new_id] = content
        self._reverse_index[content] = new_id
        return new_id

class KeywordLibrary(BaseLibrary):
    # Store keywords, deduplicate by string.
    def __init__(self):
        super().__init__()
        self._reverse_index: Dict[str, str] = {}

    def _exists(self, content: str) -> Tuple[bool, str]:
        if content in self._reverse_index:
            return True, self._reverse_index[content]
        return False, ""

    def add(self, content: str) -> str:
        if not isinstance(content, str):
            raise ValueError("KeywordLibrary content must be str.")
        exists, existing_id = self._exists(content)
        if exists:
            return existing_id
        new_id = self._generate_id()
        self._store[new_id] = content
        self._reverse_index[content] = new_id
        return new_id

class LiteratureLibrary(BaseLibrary):
    # Store literature (dict), deduplicate by pmid.
    def __init__(self):
        super().__init__()
        self._pmid_index: Dict[str, str] = {}

    def _exists(self, content: Dict[str, Any]) -> Tuple[bool, str]:
        pmid = content.get("pmid")
        if pmid and pmid in self._pmid_index:
            return True, self._pmid_index[pmid]
        return False, ""

    def add(self, content: Dict[str, Any]) -> str:
        if not isinstance(content, dict) or "pmid" not in content:
            raise ValueError("LiteratureLibrary content must be dict with 'pmid'.")
        exists, existing_id = self._exists(content)
        if exists:
            return existing_id
        new_id = self._generate_id()
        self._store[new_id] = content
        self._pmid_index[content["pmid"]] = new_id
        return new_id

class KGLibrary(BaseLibrary):
    # Store KG nodes and edges with deduplication.
    def __init__(self):
        super().__init__()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self._node_index: Dict[str, str] = {}  # entity_id -> node_id
        self._edge_index: Dict[Tuple[str, str, str], str] = {}  # (source, target, relation) -> edge_id

    def add_node(self, node_content: Dict[str, Any]) -> str:
        # Add node by unique entity_id.
        if "entity_id" not in node_content:
            raise ValueError("KGLibrary.add_node requires 'entity_id'.")
        eid = node_content["entity_id"]
        if eid in self._node_index:
            return self._node_index[eid]
        new_id = self._generate_id()
        self.nodes[new_id] = node_content
        self._node_index[eid] = new_id
        return new_id

    def add_edge(self, edge_content: Dict[str, Any]) -> str:
        # Add edge by (source, target, relation) tuple.
        if not all(k in edge_content for k in ("source", "target", "relation")):
            raise ValueError("KGLibrary.add_edge requires 'source','target','relation'.")
        key = (edge_content["source"], edge_content["target"], edge_content["relation"])
        if key in self._edge_index:
            return self._edge_index[key]
        new_id = self._generate_id()
        self.edges[new_id] = edge_content
        self._edge_index[key] = new_id
        return new_id

    def get_node(self, node_id: str) -> Dict[str, Any]:
        # Get node by id.
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Dict[str, Any]:
        # Get edge by id.
        return self.edges.get(edge_id)

    def all_nodes(self) -> Dict[str, Dict[str, Any]]:
        # Get all nodes.
        return dict(self.nodes)

    def all_edges(self) -> Dict[str, Dict[str, Any]]:
        # Get all edges.
        return dict(self.edges)

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

class EvidenceLibrary(BaseLibrary):
    # Store mapping between hypothesis and its supporting evidence.
    def __init__(self):
        super().__init__()

    def add(
        self,
        hypothesis_id: str,
        literature_ids: List[str],
        kg_node_ids: List[str],
        kg_edge_ids: List[str],
        prev_hypothesis_ids: List[str] = None
    ) -> str:
        if prev_hypothesis_ids is None:
            prev_hypothesis_ids = []
        content = {
            "hypothesis_id": hypothesis_id,
            "literature_ids": literature_ids,
            "kg_node_ids": kg_node_ids,
            "kg_edge_ids": kg_edge_ids,
            "prev_hypothesis_ids": prev_hypothesis_ids
        }
        new_id = self._generate_id()
        self._store[new_id] = content
        return new_id

    def all(self) -> Dict[str, Any]:
        # Get all evidence records.
        return dict(self._store)
