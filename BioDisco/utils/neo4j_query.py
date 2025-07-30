import os, time, re
import pickle
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
# import torch
import json
import faiss
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
import openai

from .llm_config import (
    gpt4turbo_mini_config,
    gpt4turbo_mini_config_graph,
    gpt4o_mini_config_graph
)

def clean_llm_config(llm_config):
    # Clean LLM config for OpenAI API compatibility
    if isinstance(llm_config, dict):
        cd = llm_config.copy()
        if "max_tokens" in cd:
            cd["max_output_tokens"] = cd.pop("max_tokens")
        if "max_completion_tokens" in cd:
            cd["max_output_tokens"] = cd.pop("max_completion_tokens")
        for key in ["cache_seed", "config_list"]:
            cd.pop(key, None)
        return cd
    return llm_config

def ensure_specific_llm_config(llm_config):
    # Ensure config is OpenAIGPTConfig instance
    if isinstance(llm_config, dict):
        return OpenAIGPTConfig(**llm_config)
    if type(llm_config) is LLMConfig:
        return OpenAIGPTConfig(**llm_config.dict())
    return llm_config

# Global variables
_NODE_NAMES = None
_ST_MODEL = None
_FAISS_INDEX = None
_KW_EMB_CACHE: Dict[str, np.ndarray] = {}
_MAX_JSON_CHAR = 5000
_kg_state: Dict[str, Dict[str, set]] = {}
relationship_embeddings = {
    "disease_protein": [0.9, 0.1, 0.0],
    "protein_protein": [0.8, 0.2, 0.0],
    "disease_disease": [0.8, 0.2, 0.0],
    "disease_phenotype_positive": [0.85, 0.15, 0.0],
    "drug_effect": [0.2, 0.8, 0.0],
    "contraindication": [0.1, 0.9, 0.0],
}

def LOG(msg: str):
    # Print log message with timestamp
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def init_embeddings(
    index_path: str = None,
    names_path: str = None,
    model_name: str = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
):
    # if index_path is None:
    #     # Get the directory where this file is located
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     # Go up to BioDisco package root, then to kg folder
    #     package_root = os.path.dirname(current_dir)
    #     # Go up to BioDisco package root, then to kg folder
    #     index_path = os.path.join(package_root, "kg", "node_index.faiss")
    
    # if names_path is None:
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     package_root = os.path.dirname(current_dir)
    #     names_path = os.path.join(package_root, "kg", "node_names.pkl")

    ## load path from environment variable if set
    kg_path = os.getenv("KG_PATH", None)
    if kg_path:
        index_path = os.path.join(kg_path, "node_index.faiss")
        names_path = os.path.join(kg_path, "node_names.pkl")
    else:
        raise Exception("KG_PATH environment variable not set! Please set it to the path containing node_index.faiss and node_names.pkl")

    # Initialize FAISS index, node names, and embedding model
    global _NODE_NAMES, _FAISS_INDEX, _ST_MODEL
    if _FAISS_INDEX is None:
        if os.path.exists(index_path) and os.path.exists(names_path):
            _FAISS_INDEX = faiss.read_index(index_path)
            with open(names_path, "rb") as f:
                _NODE_NAMES = pickle.load(f)
            _ST_MODEL = SentenceTransformer(model_name)
        else:
            raise FileNotFoundError("FAISS index or node names not found! Run build_kg_index.py first.")

def embed_map_keywords(raw_kws: List[str], top_k: int = 15) -> List[str]:
    # Map keywords to closest KG node names using embeddings
    LOG(f"Embedding keywords: {raw_kws}")
    init_embeddings()
    kw_embs = []
    for kw in raw_kws:
        if kw not in _KW_EMB_CACHE:
            emb = _ST_MODEL.encode([kw], convert_to_numpy=True)[0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            _KW_EMB_CACHE[kw] = emb
            LOG(f"Encoded and cached keyword: {kw}")
        kw_embs.append(_KW_EMB_CACHE[kw])
    kw_np = np.stack(kw_embs).astype("float32")
    D, I = _FAISS_INDEX.search(kw_np, top_k)
    mapped = []
    for i, idxs in enumerate(I):
        hits = [_NODE_NAMES[idx] for idx in idxs]
        LOG(f"Candidates for {raw_kws[i]}: {hits}")
        mapped.extend(hits)
    mapped_unique = list(dict.fromkeys(mapped))
    LOG(f"Final mapped keywords: {mapped_unique}")
    return mapped_unique

class FilterKeywordsAgent(ChatAgent):
    # Select most relevant KG node names based on background
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=(
                "You are a biomedical knowledge assistant.\n"
                "Task: Given a BACKGROUND text and a list of CANDIDATE knowledge-graph node ,\n"
                "Based on the background information, select the **most relevant** nodes(5-10) to use for subgraph construction.\n"
                "- Only choose from the provided candidates.\n"
                "- Output **only** a JSON array of the selected .\n"
                "- Do not output any extra text, explanations, or formatting."
                "- Output must be a valid JSON array, e.g. [\"node1\",\"node2\"]"
            )
        )
        cfg.name = "FilterKeywordsAgent"
        super().__init__(cfg)

    def filter(self, background: str, candidates: List[str]) -> List[str]:
        prompt = (
            f"{self.config.system_message}\n\n"
            f"BACKGROUND:\n{background}\n\n"
            f"CANDIDATES:\n{', '.join(candidates)}\n\n"
            "SELECTED:"
        )
        resp = self.llm_response(prompt)
        return json.loads(resp.content)

############################################################
# Domain configuration for KG querying
############################################################
DOMAIN_CONFIG = {
    "molecular": {
        "node_types": ["gene/protein","molecular_function","biological_process","cellular_component"],
        "relation_types": [
            "protein_protein","molfunc_protein","molfunc_molfunc",
            "bioprocess_protein","bioprocess_bioprocess",
            "cellcomp_protein","cellcomp_cellcomp",
            "phenotype_protein"
        ],
        "max_depth": 3, "batch_size": 10,
        "description": "Explore intracellular signaling pathways, transcription regulation, protein interactions and functional associations."
    },
    "disease": {
        "node_types": ["disease","gene/protein","biological_process","effect/phenotype","exposure"],
        "relation_types": [
            "disease_protein","protein_protein","disease_disease",
            "disease_phenotype_positive","disease_phenotype_negative",
            "phenotype_protein","phenotype_phenotype","bioprocess_protein"
        ],
        "max_depth": 3, "batch_size": 10,
        "description": "Analyze associations among diseases, molecular factors, phenotypes and pathological processes to propose new pathogenesis mechanisms."
    },
    "drug": {
        "node_types": ["drug","gene/protein","effect/phenotype","disease"],
        "relation_types": ["drug_protein","protein_protein","drug_effect","indication","contraindication","off-label use"],
        "max_depth": 3, "batch_size": 10,
        "description": "Identify relationships among drugs, molecular targets, mechanisms of action, and clinical outcomes to propose new indications or repurposing opportunities."
    },
    "systems": {
        "node_types": ["pathway","biological_process","gene/protein","molecular_function"],
        "relation_types": ["pathway_protein","pathway_pathway","protein_protein","bioprocess_protein","bioprocess_bioprocess","phenotype_protein"],
        "max_depth": 3, "batch_size": 10,
        "description": "Understand diseases and biological processes from a network perspective to explore new regulatory circuits and signaling pathways."
    },
    "structural": {
        "node_types": ["anatomy","cellular_component","gene/protein","disease"],
        "relation_types": ["anatomy_protein_present","anatomy_protein_absent","cellcomp_protein","cellcomp_cellcomp","protein_protein"],
        "max_depth": 3, "batch_size": 10,
        "description": "Explore the distribution and function of molecules in anatomical and cellular structures to propose tissue-specific regulatory mechanisms."
    },
    "environment": {
        "node_types": ["exposure","effect/phenotype","disease","molecular_function"],
        "relation_types": ["exposure_disease","exposure_protein","exposure_bioprocess","exposure_cellcomp","disease_phenotype_positive"],
        "max_depth": 3, "batch_size": 10,
        "description": "Investigate the impact of environmental factors on biological processes, gene expression and disease occurrence to explore potential links between the environment and biological states."
    }
}

############################################################
# Keyword cleaning and splitting
############################################################
def clean_and_split_keywords(raw_keywords: List[str]) -> List[str]:
    # Clean and split keywords into atomic components
    LOG(f"Original keywords: {raw_keywords}")
    cleaned = []
    for kw in raw_keywords:
        base = re.sub(
            r"\b(genes?|kinases?|pathways?|signaling|cascade)s?\b",
            "",
            kw,
            flags=re.I
        ).strip()
        LOG(f"Suffix removed: {kw} → {base}")
        parts = re.split(r"[\/,]|\s+and\s+", base)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = re.match(r"^([A-Za-z]+)(\d+)\/\d+$", p)
            if m:
                gene, num = m.groups()
                a, b = f"{gene}{num}", f"{gene}{int(num)+1}"
                LOG(f"Numeric split: {p} → [{a}, {b}]")
                cleaned.extend([a, b])
            else:
                cleaned.append(p)
    seen, final = set(), []
    for t in cleaned:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            final.append(t)
    LOG(f"Cleaned keywords: {final}")
    return final

############################################################
# Helper functions
############################################################
def simulate_embedding(text: str) -> List[float]:
    # Simulate a vector embedding for testing
    vec=[0.0]*3
    for i,c in enumerate(text[:3]):
        vec[i]=(ord(c)%10)/10.0
    return vec

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    # Compute cosine similarity between two vectors
    a, b = np.array(v1), np.array(v2)
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1==0 or n2==0: return 0.0
    return float(np.dot(a,b)/(n1*n2))

def standardize_node(name: str) -> Dict:
    # Return standardized node structure
    return {"standardized_id": f"UMLS:{abs(hash(name))%10000}", "synonyms":[name+"_synonym"]}

def generate_path_info_multi(keywords: List[str], hops: int, node_set: List[str]) -> Dict:
    # Generate multi-hop path information
    return {"hops":hops,"centrality":0.5,
            "path_summary":f"Multi-hop path with keywords [{','.join(keywords)}], nodes [{','.join(node_set)}], hops={hops}"}

def build_json_output(nodes, direct_edges, multi_edges, query_keywords, domain):
    # Build subgraph JSON for output
    json_nodes=[{"id":n["id"],"name":n["name"],"type":n["type"],
                 "attributes":standardize_node(n["name"])} for n in nodes]
    json_direct=[{"source":e["source"],"target":e["target"],"relation_type":e["relation_type"],
                  "weight":e["weight"],"path_info":e.get("path_info",{})} for e in direct_edges]
    return {"core_keywords":query_keywords,"domain":domain,"nodes":json_nodes,
            "direct_edges":json_direct,"multihop_paths":multi_edges}

############################################################
# Neo4jGraph class for KG access
############################################################
class Neo4jGraph:
    def __init__(self, uri:str, user:str, password:str):
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(uri, auth=(user,password))
    def close(self): self.driver.close()

    def _node_exists(self, kw:str)->bool:
        # Check if a node exists with given keyword
        with self.driver.session() as s:
            return s.run("MATCH (n) WHERE toLower(n.name) CONTAINS toLower($kw) RETURN n LIMIT 1", kw=kw).single() is not None

    def _get_direct(self, keyword:str, rels:List[str], batch:int):
        # Get direct neighbors and edges for keyword
        nodes, edges = {}, []
        with self.driver.session() as session:
            for r in rels:
                q = f"""MATCH (n)-[r:`{r}`]-(m)
                        WHERE toLower(n.name) CONTAINS toLower($kw)
                           OR toLower(m.name) CONTAINS toLower($kw)
                        RETURN n,r,m LIMIT $lim"""
                for rec in session.run(q, kw=keyword, lim=batch):
                    n,m,rel=rec["n"],rec["m"],rec["r"]
                    for v in (n,m):
                        vid=v.id
                        nodes[vid]={"id":vid,"name":v.get("name"),"type":list(v.labels)[0]}
                    w=cosine_similarity(relationship_embeddings.get(rel.type,[0,0,0]),[0.5,0.5,0.5])
                    edges.append({"source":n.get("name"),"target":m.get("name"),"relation_type":rel.type,
                                  "weight":w,"path_info":{"hops":1,"centrality":w,"path_summary":f"{n.get('name')}→{m.get('name')}"}})
        return nodes,edges

    def _get_multihop(self, keywords:List[str], depth:int, batch:int):
        # Get multi-hop paths for keywords
        paths=[]
        with self.driver.session() as session:
            q=f"""MATCH path=(n)-[*1..{depth}]-(m)
                  WHERE ANY(kw IN $kws WHERE ANY(x IN nodes(path) WHERE toLower(x.name) CONTAINS toLower(kw)))
                  RETURN path LIMIT $lim"""
            for rec in session.run(q,kws=keywords,lim=batch):
                p=rec["path"]; hops=len(p.relationships)
                nodeset=sorted({nd.get("name","UNK") for nd in p.nodes})
                pi=generate_path_info_multi(keywords,hops,nodeset)
                conns=[]
                for i,rel in enumerate(p.relationships):
                    a,b=p.nodes[i],p.nodes[i+1]
                    w=cosine_similarity(relationship_embeddings.get(rel.type,[0,0,0]),[0.5,0.5,0.5])
                    conns.append({"source":a.get("name"),"target":b.get("name"),"relation_type":rel.type,"weight":w})
                paths.append({"path_info":pi,"connections":conns})
        return paths

    def get_subgraph(
        self,
        background: str,
        query_keywords: List[str],
        domain: Optional[str] = None,
        relationship_types_override: Optional[List[str]] = None,
        max_depth_override: Optional[int] = None,
        hypo_id: Optional[str] = None,
        skip_mapping: bool = False
    ) -> str:
        # Build KG subgraph for given keywords and domain
        if skip_mapping:
            kws = query_keywords
        else:
            cleaned = clean_and_split_keywords(query_keywords)
            mapped = embed_map_keywords(cleaned, top_k=15)
            filter_agent = FilterKeywordsAgent()
            filtered = filter_agent.filter(background, mapped)
            LOG(f"[FilterKeywords] filtered candidates: {filtered}")
            kws = filtered
        if not kws:
            return json.dumps({"nodes": [], "direct_edges": [], "multihop_paths": []}, ensure_ascii=False)
        domains = [d.strip() for d in (domain or "").split(",") if d.strip()]
        merged = {"node_types": set(), "relation_types": set(), "max_depth": 1, "batch_size": 10}
        for d in domains:
            cfg = DOMAIN_CONFIG.get(d, {})
            merged["node_types"] |= set(cfg.get("node_types", []))
            merged["relation_types"] |= set(cfg.get("relation_types", []))
            merged["max_depth"] = max(merged["max_depth"], cfg.get("max_depth", 1))
            merged["batch_size"] = max(merged["batch_size"], cfg.get("batch_size", 10))
        if "disease" in domains:
            merged["relation_types"] |= {"disease_protein", "protein_protein"}
        domain_cfg = {
            "node_types": list(merged["node_types"]),
            "relation_types": list(merged["relation_types"]),
            "max_depth": merged["max_depth"],
            "batch_size": merged["batch_size"]
        }
        rels  = relationship_types_override or domain_cfg["relation_types"]
        depth = max_depth_override or domain_cfg["max_depth"]
        batch = domain_cfg["batch_size"]
        valid = [kw for kw in kws if self._node_exists(kw)]
        if not valid:
            return json.dumps({"nodes": [], "direct_edges": [], "multihop_paths": []}, ensure_ascii=False)
        all_nodes, all_edges = {}, []
        for kw in valid:
            nd, ed = self._get_direct(kw, rels, batch)
            all_nodes.update(nd)
            all_edges.extend(ed)
        existing_names = {n["name"].lower() for n in all_nodes.values()}
        with self.driver.session() as session:
            for orig in query_keywords:
                q1 = "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($kw) RETURN n LIMIT 3"
                for rec in session.run(q1, kw=orig):
                    v, vid = rec["n"], rec["n"].id
                    if vid not in all_nodes:
                        all_nodes[vid] = {"id": vid, "name": v.get("name"), "type": list(v.labels)[0]}
                        existing_names.add(v.get("name").lower())
                for tok in re.findall(r"\b[\w\-]+\b", orig):
                    lt = tok.lower()
                    if any(lt in nm for nm in existing_names):
                        continue
                    q2 = "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($tok) RETURN n LIMIT 3"
                    for rec in session.run(q2, tok=tok):
                        v, vid = rec["n"], rec["n"].id
                        if vid not in all_nodes:
                            all_nodes[vid] = {"id": vid, "name": v.get("name"), "type": list(v.labels)[0]}
                            existing_names.add(v.get("name").lower())
        if "disease" in domains:
            has_dp = any(e["relation_type"] == "disease_protein" for e in all_edges)
            has_pp = any(e["relation_type"] == "protein_protein" for e in all_edges)
            if not has_dp or not has_pp:
                for kw in valid:
                    for r in ["disease_protein", "protein_protein"]:
                        nd, ed = self._get_direct(kw, [r], batch)
                        all_nodes.update(nd)
                        all_edges.extend(ed)
        mh_raw = self._get_multihop(valid, depth, batch)
        agg, mh_paths = {}, []
        for rec in mh_raw:
            key = rec["path_info"]["path_summary"]
            if key not in agg:
                agg[key] = True
                mh_paths.append(rec)
        candidate = build_json_output(
            list(all_nodes.values()), all_edges, mh_paths, kws, domain
        )
        s = json.dumps(candidate, ensure_ascii=False)
        if len(s) > _MAX_JSON_CHAR and (relationship_types_override or max_depth_override):
            defs = DOMAIN_CONFIG.get(domains[0], {})
            rd, dp = defs.get("relation_types", []), defs.get("max_depth", 1)
            nd2, ed2 = {}, []
            for kw in valid:
                n2, e2 = self._get_direct(kw, rd, batch)
                nd2.update(n2)
                ed2.extend(e2)
            mh2 = self._get_multihop(valid, dp, batch)
            ed2s = sorted(ed2, key=lambda e: e["weight"], reverse=True)[:10]
            mh2f = [m for m in mh2 if m["path_info"]["centrality"] >= 0.5][:5]
            candidate = build_json_output(list(nd2.values()), ed2s, mh2f, valid, domain)
        if hypo_id:
            state = _kg_state.setdefault(hypo_id, {"seen_nodes": set(), "seen_edges": set(), "seen_multi": set()})
            new_nodes, new_edges, new_multi = [], [], []
            for n in candidate["nodes"]:
                if n["id"] not in state["seen_nodes"]:
                    state["seen_nodes"].add(n["id"])
                    new_nodes.append(n)
            for e in candidate["direct_edges"]:
                key = (e["source"], e["target"], e["relation_type"])
                if key not in state["seen_edges"]:
                    state["seen_edges"].add(key)
                    new_edges.append(e)
            for m in candidate["multihop_paths"]:
                ps = m["path_info"]["path_summary"]
                if ps not in state["seen_multi"]:
                    state["seen_multi"].add(ps)
                    new_multi.append(m)
            candidate["nodes"], candidate["direct_edges"], candidate["multihop_paths"] = (
                new_nodes, new_edges, new_multi
            )
        return json.dumps(candidate, ensure_ascii=False, indent=2)

def get_node_type_from_list(node_list, node_name):
    # Get node type by node name
    for n in node_list:
        if n["name"]==node_name:
            return n.get("type","Unknown")
    return "Unknown"

def build_readable_kg_context(nodes, direct_edges, multihop_paths):
    # Build a human-readable KG context string
    name2type={n["name"]:n.get("type","Unknown") for n in nodes}
    for path in multihop_paths:
        for c in path["connections"]:
            for nm in (c["source"],c["target"]):
                if nm not in name2type:
                    name2type[nm]="gene/protein"
    node_lines=[f"{name}({tp})" for name, tp in name2type.items()]
    edges=[]
    for e in direct_edges:
        edges.append(f"{e['source']}({name2type.get(e['source'])})→{e['relation_type']}({e['weight']:.3f})→{e['target']}({name2type.get(e['target'])})")
    paths=[]
    for path in multihop_paths:
        parts=[]
        for h in path["connections"]:
            parts.append(f"{h['source']}({name2type.get(h['source'])})")
            parts.append(f"{h['relation_type']}({h['weight']:.3f})")
        tgt=path["connections"][-1]["target"]
        parts.append(f"{tgt}({name2type.get(tgt)})")
        paths.append("→".join(parts))
    return "Nodes:\n" + ", ".join(node_lines) + "\n\nDirectEdges:\n" + "\n".join(edges) + "\n\nMultiHopPaths:\n" + "\n".join(paths)
