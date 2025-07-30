import sys, re, json
import os
from itertools import combinations
from typing import Optional, List, Tuple, Union, Dict, Any
from pydantic import BaseModel
from langroid.language_models.base import LLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import random
from datetime import datetime, timedelta
from pathlib import Path

from .utils.log_utils import write_agent_log
from .utils.llm_config import (
    gpt4turbo_mini_config,
    gpt4turbo_mini_config_graph,
    gpt4o_mini_config_graph
)
from .utils.neo4j_query import Neo4jGraph, build_readable_kg_context, DOMAIN_CONFIG, FilterKeywordsAgent, clean_and_split_keywords, embed_map_keywords
from .utils.pubmed_query import (
    KeywordQueryAgent, 
    HypothesisQueryAgent, 
    adaptive_pubmed_search
)
from .utils.libraries import HypothesisLibrary

hypo_lib = HypothesisLibrary()

all_kg_nodes_set: set = set()
all_kg_edges_set: set = set()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

neo4j_graph = None
def get_neo4j_graph():
    """Get Neo4j graph instance, creating it if necessary"""
    global neo4j_graph
    if neo4j_graph is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, password]):
            raise Exception(
                "Neo4j credentials not found. Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables."
            )
        
        neo4j_graph = Neo4jGraph(uri=uri, user=user, password=password)
    
    return neo4j_graph

# neo4j_graph = Neo4jGraph(
#     uri=NEO4J_URI,
#     user=NEO4J_USER,
#     password=NEO4J_PASSWORD
# )

# Parse hypotheses and evidence from raw output
def parse_hypos(raw: str):
    pattern = re.compile(r'([^\n]+?)\s*EVIDENCE:([^\n]+)')
    hypos, evs = [], []
    for m in pattern.finditer(raw):
        hypos.append(m.group(1).strip())
        evs.append(m.group(2).strip())
    return hypos, evs

# Parse refined hypothesis and evidence
def parse_refined(ref_raw):
    lines = [l.strip() for l in ref_raw.splitlines() if l.strip()]
    if not lines:
        return None, None
    last = lines[-1]
    if "NEW_EVIDENCE:" in last and not last.startswith("NEW_EVIDENCE:"):
        hypo_part, ev_part = last.split("NEW_EVIDENCE:", 1)
        hypo = hypo_part.strip().removeprefix("Refined hypothesis:").strip()
        ev   = "NEW_EVIDENCE:" + ev_part.strip()
        return hypo, ev
    if last.startswith("EVIDENCE:") or last.startswith("NEW_EVIDENCE:"):
        hypo = None
        for l in reversed(lines[:-1]):
            if not (l.startswith("EVIDENCE:") or l.startswith("NEW_EVIDENCE:")):
                hypo = l.removeprefix("Refined hypothesis:").strip()
                break
        ev = last
        return hypo, ev
    return lines[-1].removeprefix("Refined hypothesis:").strip(), ""

# Parse hypotheses with evidence fields
def parse_hypos_with_evidence(sci_raw):
    hypos, evidences = [], []
    for line in sci_raw.splitlines():
        line = line.strip()
        if not line: continue
        if "EVIDENCE:" in line:
            hypo, evidence = line.split("EVIDENCE:", 1)
            hypos.append(hypo.strip())
            evidences.append("EVIDENCE:" + evidence.strip())
        else:
            hypos.append(line)
            evidences.append("EVIDENCE:")
    return hypos, evidences

# Merge two evidence fields, keeping all unique values
def merge_evidence(prior_evidence: str, new_evidence: str):
    import re
    from collections import OrderedDict
    field_order = ['LIT', 'KGNODE', 'KGEDGE']
    all_fields = OrderedDict((k, set()) for k in field_order)
    def parse_ev(ev):
        result = {k: set() for k in field_order}
        if not ev:
            return result
        ev = ev.strip()
        for prefix in ["EVIDENCE:", "NEW_EVIDENCE:"]:
            if ev.startswith(prefix):
                ev = ev[len(prefix):]
        parts = re.split(r',\s*(?=[A-Z]+:)', ev)
        for part in parts:
            for k in field_order:
                if part.startswith(k + ":"):
                    content = part[len(k)+1:].strip()
                    if content and content.upper() != "NONE":
                        vals = [v.strip() for v in content.split(",") if v.strip() and v.strip().upper() != "NONE"]
                        result[k].update(vals)
        return result
    prior_dict = parse_ev(prior_evidence)
    new_dict = parse_ev(new_evidence)
    merged = OrderedDict()
    for k in field_order:
        if not new_dict[k]:
            merged[k] = set(prior_dict[k])
        else:
            merged[k] = set(prior_dict[k]) | set(new_dict[k])
    ev_strs = []
    for k in field_order:
        if merged[k]:
            ev_strs.append(f"{k}:{','.join(sorted(merged[k], key=str.lower))}")
    if ev_strs:
        return "EVIDENCE:" + ",".join(ev_strs)
    else:
        return ""

# Extract PMID, title, abstract from PubMed JSON
def extract_pubmed_info(pubmed_json_str, max_n=8):
    result = []
    if not pubmed_json_str.strip():
        return result
    for line in pubmed_json_str.strip().splitlines():
        try:
            entry = json.loads(line)
            pmid = str(entry.get("id") or entry.get("pmid") or "")
            title = entry.get("title") or ""
            abstract = entry.get("abstract") or entry.get("summary") or ""
            if pmid and title and abstract:
                result.append({"pmid": pmid, "title": title, "abstract": abstract})
            if len(result) >= max_n:
                break
        except Exception as e:
            continue
    return result

# Extract all metric blocks from CriticAgent output
def extract_metric_blocks(feedback: str):
    pattern = re.compile(
        r'\*\*(.*?)\*\*:\s*Score\s*(\d{1,2})\s*[\--—:]?\s*(.*?)'
        r'(?=(\n\*\*|\n?\|\s*Metric|\nOverall Score:|\Z))',
        re.DOTALL)
    return [(mname.strip(), int(score), txt.strip()) for mname, score, txt, _ in pattern.findall(feedback)]

# Clean llm_config by removing unnecessary keys
def clean_llm_config(llm_config):
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

# Convert llm_config to OpenAIGPTConfig if needed
def ensure_specific_llm_config(llm_config):
    if isinstance(llm_config, dict):
        return OpenAIGPTConfig(**llm_config)
    if type(llm_config) is LLMConfig:
        return OpenAIGPTConfig(**llm_config.dict())
    return llm_config

# DomainSelectorAgent: select relevant domains based on background embedding
class DomainSelectorAgent(ChatAgent):
    EMBEDDING_MODEL_NAME = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
    THRESHOLD_RATIO = 0.7
    MAX_DOMAINS = 3

    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=(
                f"DomainSelectorAgent:\n"
                f"Task: Identify the most relevant research domain(s) for the given biomedical background text.\n"
                f"Return up to {self.MAX_DOMAINS} domains separated by commas."
            )
        )
        cfg.name = "DomainSelectorAgent"
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        self.domain_keys = list(DOMAIN_CONFIG.keys())
        descriptions = [DOMAIN_CONFIG[key]["description"] for key in self.domain_keys]
        self.domain_embeddings = self.embed_model.encode(
            descriptions, convert_to_tensor=True
        )

    def step(self, background: str) -> str:
        import torch
        bg_emb = self.embed_model.encode(background, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(
            bg_emb.unsqueeze(0), self.domain_embeddings, dim=1
        )
        sims_list = sims.tolist()
        top_score = max(sims_list)
        selected = [
            dom for dom, sim in zip(self.domain_keys, sims_list)
            if sim >= top_score * self.THRESHOLD_RATIO
        ]
        if not selected:
            idx = sims_list.index(top_score)
            selected = [self.domain_keys[idx]]
        selected = selected[:self.MAX_DOMAINS]
        if "disease" not in selected:
            selected.insert(0, "disease")
        if "molecular" not in selected:
            selected.insert(0, "molecular")
        selected = selected[:self.MAX_DOMAINS]
        return ",".join(selected)

# Query Neo4j KG for subgraph given background and keywords
def call_neo4j_subgraph_core(
    background: str,
    keywords: Union[str, List[str]],
    important_rel_types: Optional[List[str]] = None,
    domain: str = None,
    max_depth_override: Optional[int] = None,
    hypo_id: Optional[str] = None,
    direct_edge_limit: Optional[int] = None,
    node_limit: Optional[int] = None
) -> str:
    if isinstance(keywords, str):
        initial_kws = [keywords]
    else:
        initial_kws = keywords
    cleaned = clean_and_split_keywords(initial_kws)
    mapped  = embed_map_keywords(cleaned, top_k=15)
    candidates = list(dict.fromkeys(mapped))
    if not candidates:
        return json.dumps({"nodes": [], "direct_edges": [], "multihop_paths": []}, ensure_ascii=False)
    filter_agent   = FilterKeywordsAgent()
    filtered_kws   = filter_agent.filter(background, candidates)
    if not filtered_kws:
        return json.dumps({"nodes": [], "direct_edges": [], "multihop_paths": []}, ensure_ascii=False)
    
    graph = get_neo4j_graph()
    summary_str = graph.get_subgraph(
        background=background,
        query_keywords=filtered_kws,
        domain=domain,
        relationship_types_override=important_rel_types,
        max_depth_override=max_depth_override,
        hypo_id=hypo_id,
        skip_mapping=True
    )
    summary = json.loads(summary_str)
    if direct_edge_limit is not None and summary.get("direct_edges"):
        summary["direct_edges"].sort(key=lambda e: e.get("weight", 0), reverse=True)
        summary["direct_edges"] = summary["direct_edges"][:direct_edge_limit]
    if summary.get("direct_edges"):
        kept = {e["source"] for e in summary["direct_edges"]} | {e["target"] for e in summary["direct_edges"]}
        filtered_nodes = [n for n in summary.get("nodes", []) if n["name"] in kept]
    else:
        filtered_nodes = summary.get("nodes", [])
    if node_limit is not None:
        filtered_nodes = filtered_nodes[:node_limit]
    summary["nodes"] = filtered_nodes
    kg_context = build_readable_kg_context(
        summary["nodes"],
        summary["direct_edges"],
        summary["multihop_paths"]
    )
    return kg_context

# Switch for disabling KG calls
# DISABLE_KG = False
def call_neo4j_subgraph(*args, **kwargs) -> str:
    DISABLE_KG = os.getenv("DISABLE_KG", "true").lower() in {"true"}
    print('disable_kg:', DISABLE_KG)
    if DISABLE_KG:
        return ""
    return call_neo4j_subgraph_core(*args, **kwargs)

# PubMed search agent with LLM-based strategy
# 1. 定义真正的 PubMed 查询函数
def call_pubmed_search_core(
    keywords: List[str],
    hypothesis: Optional[str] = None,
    feedback: Optional[str] = None,
    min_results: int = 1,
    max_results: int = 10,
    background: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    # ...（你原来的实现）...
    if hypothesis and feedback:
        prompt = f"Hypothesis: {hypothesis}\nLow Score Feedback: {feedback}\nKeywords: {', '.join(keywords)}"
        agent = HypothesisQueryAgent()
        strategy = agent.get_strategy(prompt)
    else:
        agent = KeywordQueryAgent()
        strategy = agent.get_strategy(", ".join(keywords))
    articles, used_groups, status = adaptive_pubmed_search(
        strategy,
        field_pref="MeSH/TIAB",
        min_results=min_results,
        max_results=max_results,
        end_date=end_date
    )
    return {
        "strategy": strategy,
        "status": status,
        "articles": articles,
        "used_groups": used_groups
    }

# 2. 切换开关函数
# DISABLE_PUBMED = False

def call_pubmed_search(*args, **kwargs):
    DISABLE_PUBMED = os.getenv("DISABLE_PUBMED", "true").lower() in {"true"}
    print('disable_pubmed:', DISABLE_PUBMED)
    if DISABLE_PUBMED:
        return {
            "strategy": {},
            "status": "PubMed search disabled",
            "articles": [],
            "used_groups": []
        }
    return call_pubmed_search_core(*args, **kwargs)


# KeywordExtractorAgent: extract biomedical entities using BioBERT NER or LLM
class KeywordExtractorAgent(ChatAgent):
    NER_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
    _tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    _model     = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    ner_pipeline = pipeline(
        "ner",
        model=_model,
        tokenizer=_tokenizer,
        aggregation_strategy="simple"
    )
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=(
                "KeywordExtractorAgent:\n"
                "Task: From the input biomedical text, extract **only canonical biomedical entities**—"
                "genes, proteins, diseases, drugs, small molecules, cell types, pathways, phenotypes, etc.  \n"
                "If there are common terms in the phrase, such as signal, signaling, disease, progress, cell, activity, etc., please delete them and keep only the most concise biological entity name.\n"
                "- If you truly find no biomedical entity, you may output up to 3 fallback noun phrases.  \n"
                "- Output as a comma-separated list of **3–8** entity names, no duplicates."
            )
        )
        cfg.name = "KeywordExtractorAgent"
        super().__init__(cfg)
        self.stop_words = {
            "based", "information", "relationship", "the", "activity",
            "cell", "cells", "effect", "effects", "result"
        }
    def extract(self, text: str) -> List[str]:
        ents = []
        for ent in self.ner_pipeline(text):
            label = ent.get("entity_group", ent.get("entity"))
            if label in {"CHEMICAL", "DISEASE", "GENE_OR_PROTEIN", "PROTEIN", "PATHWAY"}:
                ents.append(ent["word"])
        seen = set()
        entities = []
        for w in ents:
            lw = w.lower()
            if lw not in seen:
                seen.add(lw)
                entities.append(w)
        if len(entities) >= 3:
            return entities[:8]
        prompt = f"{self.config.system_message}\n\nText:\n{text}\n\nEntities:"
        resp = self.llm_response(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        raw = [t.strip() for t in content.split(",") if t.strip()]
        filtered = [t for t in raw if not any(s in t.lower() for s in self.stop_words)]
        seen_f = set()
        unique = []
        for t in filtered:
            lt = t.lower()
            if lt not in seen_f:
                seen_f.add(lt)
                unique.append(t)
        max_k = min(12, max(3, len(text) // 50))
        if len(unique) > max_k:
            unique = unique[:max_k]
        elif len(unique) < 3:
            extras = re.findall(r"\b[a-zA-Z0-9\-]{4,}\b", text)
            for tok in extras:
                lt = tok.lower()
                if lt not in seen_f:
                    unique.append(tok)
                    seen_f.add(lt)
                if len(unique) >= 3:
                    break
        return unique

class DiseaseExplorerAgent(ChatAgent):
    def step(self, background: str, domain: str,
             direct_edge_limit=30, node_limit=30, **kwargs) -> str:
        kws = KeywordExtractorAgent().extract(background)
        query_input = kws if len(kws) > 1 else (kws[0] if kws else background.split()[0])
        return call_neo4j_subgraph(
            background=background,
            keywords=query_input,
            domain=domain,
            direct_edge_limit=direct_edge_limit,
            node_limit=node_limit
        )

class KGAgent(ChatAgent):
    def step(self, hypothesis: str, domain: str,
             depth_override=None, rels_override=None,
             direct_edge_limit=30, node_limit=30, **kwargs) -> str:
        kws = KeywordExtractorAgent().extract(hypothesis)
        return call_neo4j_subgraph(
            background=hypothesis,
            keywords=kws or [hypothesis],
            important_rel_types=rels_override,
            domain=domain,
            max_depth_override=depth_override,
            direct_edge_limit=direct_edge_limit,
            node_limit=node_limit
        )

 # PlannerAgent: generate stepwise research workflow

class PlannerAgent(ChatAgent):
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=(
                "PlannerAgent:\n"
                "Task: Develop a clear, stepwise research workflow based on the provided background text."
                " Your plan should outline:"
                " 1. Domain selection;"
                " 2. Knowledge graph retrieval steps;"
                " 3. Hypothesis generation;"
                " 4. Iterative refinement using literature and graph evidence;"
                " 5. Final decision-making."
                " Respond with numbered steps."
            )
        )
        cfg.name = "PlannerAgent"
        super().__init__(cfg)

    def step(self, background: str) -> str:
        prompt = f"{self.config.system_message}\nBackground:\n{background}\nPlan:"
        resp = self.llm_response(prompt)
        return resp.content if hasattr(resp, 'content') else str(resp)

# ScientistAgent: generate one concise, testable, and innovative biomedical hypothesis
class ScientistAgent(ChatAgent):
    def __init__(self):
        system_message = (
            "ScientistAgent\n"
            "Inputs:\n"
            "  • Background text\n"
            "  • KG context (with entities and mechanistic relationships)\n\n"
            "Task:\n"
            "Generate one concise, testable biomedical hypothesis. The hypothesis must be novel but strictly grounded in the background, reasoning from the background, and may reference the KG context.\n\n"
            "Guidelines:\n"
            "• Use both background and KG context; if possible, integrate new insights from the KG, but do not stray from the background.\n"
            "• Propose a new mechanistic, causal, regulatory, or predictive relationship—not just a restatement or paraphrase of the input.\n"
            "• Use precise scientific language and include at least one mechanistic verb (e.g., activates, inhibits, modulates, induces, represses, enhances, suppresses, regulates, predicts, promotes).\n"
            "• Do not copy, summarize, or paraphrase inputs. Only extend or refine based on them, always staying within the context of the background.\n"
            "• Write a single, scientifically plausible, and testable sentence, naming relevant entities and outcomes.\n"
            "• Do not add evidence fields, numbering, bullets, explanations, or citations. Output one line only.\n\n"
            "Examples (good):\n"
            "The inhibition of Wnt signaling reduces cardiac fibroblast activation, thereby limiting fibrosis progression in cardiovascular disease.\n"
            "Activation of TGF-β pathway in vascular smooth muscle cells promotes pathological remodeling in hypertension.\n"
            "Loss of gene X enhances the inflammatory response to environmental toxin Y in liver tissue.\n\n"
            "Examples (bad):\n"
            "• Summarizing background: 'Cardiovascular disease is associated with fibrosis and Wnt signaling.'\n"
            "• Adding evidence fields: 'The inhibition of Wnt signaling reduces fibrosis. EVIDENCE: KGNODE:Wnt,KGEDGE:Wnt-signaling→fibrosis'\n"
            "• Numbering: '1) Inhibition of Wnt signaling reduces fibrosis.'\n\n"
            "Output:\n"
            "One line: a single, original hypothesis, no numbering, no evidence fields, no extra text."
        )

        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(
                clean_llm_config(gpt4turbo_mini_config_graph)
            ),
            system_message=system_message,
        )
        cfg.name = "ScientistAgent"
        super().__init__(cfg)

    def step(self, background: str, kg_context: str) -> str:
        prompt = (
            f"{self.config.system_message}\n\n"
            f"Background:\n{background}\n\n"
            f"KG Context:\n{kg_context}\n\n"
            "Hypotheses (one line):"
        )
        resp = self.llm_response(prompt)
        return resp.content.strip()

# PubmedAgent: search PubMed for relevant articles
class PubmedAgent(ChatAgent):
    def __init__(self, min_results=1, max_results=4):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4o_mini_config_graph)),
            system_message="You are a PubMed search assistant."
        )
        cfg.name = "PubmedAgent"
        self.min_results = min_results
        self.max_results = max_results
        super().__init__(cfg)

    def step(self, keywords: List[str], hypothesis: Optional[str] = None, feedback: Optional[str] = None,
             min_results=None, max_results=None, **kwargs) -> str:
        min_results = min_results if min_results is not None else self.min_results
        max_results = max_results if max_results is not None else self.max_results
        res = call_pubmed_search(
            keywords=keywords,
            hypothesis=hypothesis,
            feedback=feedback,
            min_results=min_results,
            max_results=max_results
        )
        articles = res["articles"]
        if not articles:
            return "[PubMed] No sufficient relevant articles found."
        out_blocks = [json.dumps(a, ensure_ascii=False) for a in articles]
        return "\n".join(out_blocks)

# CriticAgent: evaluate hypothesis by 4 metrics, output scores and rationales
class CriticAgent(ChatAgent):
    def __init__(self):
        system_message = (
            "CriticAgent\n"
            "You are an expert in biomedicine. Assess the hypothesis using four metrics:\n"
            "  • Novelty\n"
            "  • Relevance\n"
            "  • Significance\n"
            "  • Verifiability\n\n"
            "For each metric, use a 0-5 scale. Provide a one-sentence rationale per metric.\n"
            "Output:\n"
            "For each metric, output in the following format (start each section with a blank line):\n"
            "**Novelty**: Score <X>\n<one-sentence rationale>\n\n"
            "**Relevance**: Score <X>\n<one-sentence rationale>\n\n"
            "**Significance**: Score <X>\n<one-sentence rationale>\n\n"
            "**Verifiability**: Score <X>\n<one-sentence rationale>\n\n"
            "At the end, on its own line, write “Overall Score: <value>/20”."
        )
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4o_mini_config_graph)),
            system_message=system_message,
        )
        cfg.name = "CriticAgent"
        super().__init__(cfg)

    def step(self, background: str, literature_info: str, hypothesis: str) -> str:
        prompt = (
            f"{self.config.system_message}\n\n"
            f"Background:\n{background}\n\n"
            f"Literature Evidence:\n{literature_info}\n\n"
            f"Hypothesis:\n{hypothesis}\n\n"
            "Provide your critique, complete the table, and then write “Overall Score: <value>/20”."
        )
        resp = self.llm_response(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

# RevisionAgent: suggest evidence source and KG settings for low metrics
class RevisionAgent(ChatAgent):
    def __init__(self, pubmed_min_results=2, pubmed_max_results=5):
        system_message = (
            "RevisionAgent:\n"
            "Task: You receive:\n"
            "  • CriticAgent's markdown critique with scores (0-5) and rationales.\n"
            "  • The current hypothesis and background text.\n\n"
            "Steps:\n"
            "1. Identify all metric(s) scoring ≤ 3. If all metrics are above 3, select one 4-point metric in this priority: Novelty > Significance > Relevance > Verifiability.\n"
            "2. For each selected metric, recommend one or more actions from ['neo4j', 'pubmed', 'background'] as follows:\n"
            "   - For low novelty or missing mechanistic details, prioritize 'neo4j' to explore new biological pathways or regulatory connections. Also consider 'pubmed' if relevant literature may offer additional insights or cross-disciplinary mechanisms.\n"
            "   - For low verifiability or weak experimental/clinical support, prioritize 'pubmed' for literature evidence; also use 'neo4j' if the knowledge graph may provide direct links to measurable outcomes or experimental readouts.\n"
            "   - For low relevance or significance, select 'background' to re-anchor the hypothesis in the biomedical context. If needed, combine with 'neo4j' or 'pubmed' to bring in additional evidence or mechanistic depth.\n"
            "   - If multiple metrics are low, recommend all relevant actions; do not limit to one.\n"
            "   - If all metrics are high (≥4), output both 'neo4j' and 'pubmed' to encourage further hypothesis enhancement.\n"
            "3. ALSO suggest how to adjust the next KG query:\n"
            "   - DEPTH_OVERRIDE: If relevant, increase max_depth by 1; otherwise keep the current value.\n"
            "   - RELS_OVERRIDE: Pick the 2–8 most relevant relation types from the global DOMAIN_CONFIG across all domains. Choose types that best address the weaknesses identified (e.g., mechanism-related for novelty gaps, evidence-related for verifiability gaps).\n"
            "4. Return exactly three lines in this format:\n"
            "   ACTIONS:action1,action2\n"
            "   DEPTH_OVERRIDE:<integer>\n"
            "   RELS_OVERRIDE:rel1,rel2,...\n"
        )
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=system_message
        )
        cfg.name = "RevisionAgent"
        self.pubmed_min_results = pubmed_min_results
        self.pubmed_max_results = pubmed_max_results
        super().__init__(cfg)

    def simple_analysis(self, metric_name: str, score: int, rationale: str, hypo: str) -> str:
        # Return concise improvement suggestion for low-scoring metric
        prompt = (
            f"Given a hypothesis:\n{hypo}\n\n"
            f"The metric '{metric_name}' was rated {score} with the following rationale:\n"
            f"{rationale}\n\n"
            "In one sentence, concisely describe what's missing or weak, and give a clear, actionable improvement suggestion."
        )
        resp = self.llm_response(prompt)
        return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    def step(
        self,
        critic_feedback: str,
        hypothesis: str,
        domain: str,
        background: str,
        direct_edge_limit: int = 30,
        node_limit: int = 30
    ) -> Tuple[
        List[str],
        List[Tuple[str, str]],
        List[str],
        Optional[int],
        Optional[List[str]]
    ]:
        import random
        metric_texts = []
        analysis_texts = []
        blocks = extract_metric_blocks(critic_feedback)
        for name, score, txt in blocks:
            if score <= 3:
                metric_texts.append(f"**{name}**: Score {score}\n{txt}")
                analysis = self.simple_analysis(name, score, txt, hypothesis)
                analysis_texts.append(f"{name}analyze：{analysis}")
        if not metric_texts and blocks:
            cands = [b for b in blocks if b[1] == 4] or blocks
            name, score, txt = random.choice(cands)
            metric_texts.append(f"**{name}**: Score {score}\n{txt}")
            analysis = self.simple_analysis(name, score, txt, hypothesis)
            analysis_texts.append(f"{name}analyze：{analysis}")
        if not metric_texts:
            metric_texts = [critic_feedback.strip()]
            analysis_texts = ["There are no low scores, and it is recommended to improve the innovativeness and verifiability of the assumptions in a global manner."]
        all_rels = sorted({rel
                           for cfg in DOMAIN_CONFIG.values()
                           for rel in cfg.get("relation_types", [])})
        prompt = (
            f"{self.config.system_message}\n\n"
            f"Critic Feedback:\n{critic_feedback}\n\n"
            f"Current Hypothesis:\n{hypothesis}\n\n"
            f"Background:\n{background}\n\n"
            f"AllRelationTypes:{','.join(all_rels)}\n"
            "Decision:"
        )
        resp = self.llm_response(prompt)
        out = resp.content if hasattr(resp, "content") else str(resp)
        actions: List[str] = []
        depth_override: Optional[int] = None
        rels_override: Optional[List[str]] = None
        for line in out.splitlines():
            line = line.strip()
            if line.upper().startswith("ACTIONS:"):
                actions = [a.strip().lower() for a in line.split(":",1)[1].split(",") if a.strip()]
            elif line.upper().startswith("DEPTH_OVERRIDE:"):
                try:
                    depth_override = int(line.split(":",1)[1].strip())
                except ValueError:
                    depth_override = None
            elif line.upper().startswith("RELS_OVERRIDE:"):
                rels_override = [r.strip() for r in line.split(":",1)[1].split(",") if r.strip()]
        if not actions:
            actions = ["neo4j"]
        if "neo4j" not in actions:
            depth_override = None
            rels_override   = None
        analysis_part = "\n".join(analysis_texts)
        new_background = f"{hypothesis}\n\n[Revision suggestions]\n{analysis_part}\n\n  Background：\n{background}"
        info_bundle: List[Tuple[str,str]] = []
        kws = None
        if "neo4j" in actions or "pubmed" in actions:
            kws = KeywordExtractorAgent().extract(hypothesis)
        if "neo4j" in actions:
            kg_dyn = call_neo4j_subgraph(
                new_background,
                keywords=kws or [hypothesis],
                domain=domain,
                important_rel_types=rels_override,
                max_depth_override=depth_override,
                direct_edge_limit=direct_edge_limit,
                node_limit=node_limit
            )
            write_agent_log(
                "Re_KGagent",
                {
                    "background": new_background,
                    "keywords": kws or [hypothesis],
                    "domain": domain,
                    "rels_override": rels_override,
                    "depth_override": depth_override
                },
                kg_dyn
            )
            info_bundle.append(("neo4j", kg_dyn))
        if "pubmed" in actions:
            res = call_pubmed_search(
                keywords=kws,
                hypothesis=hypothesis,
                feedback="\n".join(analysis_texts),
                min_results=self.pubmed_min_results,
                max_results=self.pubmed_max_results,
                background=new_background,
            )
            write_agent_log(
                "Re_PubmedAgent",
                {
                    "keywords": kws,
                    "hypothesis": hypothesis,
                    "feedback": "\n".join(analysis_texts),
                    "background": new_background
                },
                res
            )
            articles = res["articles"]
            info_bundle.append(("pubmed", "\n".join(json.dumps(a, ensure_ascii=False) for a in articles)))
        info_bundle.append(("background", new_background))
        return actions, info_bundle, metric_texts, depth_override, rels_override

# RefineAgent: refine hypothesis based on feedback and new info
class RefineAgent(ChatAgent):
    def __init__(self):
        system_message = (
            "RefineAgent\n"
            "You receive:\n"
            "  • Critic feedback (explicit weaknesses for each metric)\n"
            "  • Current hypothesis\n"
            "  • New information (from NEO4J, PUBMED, or BACKGROUND)\n\n"
            "Instructions:\n"
            "1. For each weakness identified by the Critic, briefly state what is missing or unclear in the hypothesis (one sentence per metric).\n"
            "2. Review all new information:\n"
            "   - If high-quality, relevant content directly addresses a weakness, use it as inspiration to revise the hypothesis. However, you must synthesize a new idea or mechanistic relationship that is **not a verbatim or near-verbatim copy** of any provided text.\n"
            "   - When revising, you **should introduce at least one original element**, such as a novel combination of entities, an unreported regulatory relationship, or a creative extension or integration of known pathways. Do not simply summarize or restate the background or literature.\n"
            "   - Justify any extension by clear scientific reasoning based on the available context, ensuring plausibility. You may refer to reliable domain knowledge to propose a **potential new link, prediction, or hypothesis** that is not explicitly stated in the background.\n"
            "***\n"
            "   - If there is no relevant or high-quality new information (from NEO4J, PUBMED, or BACKGROUND), **do not fabricate or add unsupported new knowledge**. In such cases, it is acceptable to leave the hypothesis unchanged and explicitly state that no substantive improvement is possible due to lack of evidence.\n"
            "***\n"
            "3. Only when revision is possible and justified by new evidence or information, do **not** merely paraphrase, summarize, or rearrange background or literature content. **Do not copy sentences, phrases, or main conclusions directly from the provided material.**\n"
            "4. Substantive content changes and increased scientific novelty are only required if there is sufficient evidence or support for a meaningful revision; otherwise, it is acceptable to keep the original hypothesis unchanged.\n"
            "Output:\n"
            "Write ONLY the final refined hypothesis (one line, ≤ 45 words, no explanation or evidence). The hypothesis must be both scientifically plausible and more innovative than the previous version if a meaningful revision is possible."
        )

        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config_graph)),
            system_message=system_message
        )
        cfg.name = "RefineAgent"
        super().__init__(cfg)

    def step(
        self,
        hypothesis: str,
        new_info_list: list,
        critic_metrics_texts: list,
        prior_evidence: str
    ) -> str:
        info_section = "\n".join(f"[{lbl}]\n{cnt}" for lbl, cnt in new_info_list)
        metrics_block = "\n\n".join(critic_metrics_texts)
        prompt = (
            f"{self.config.system_message}\n\n"
            f"Metrics To Address:\n{metrics_block}\n\n"
            f"Current Hypothesis:\n{hypothesis}\n\n"
            f"New Information:\n{info_section}\n\n"
            "Refinement:"
        )
        resp = self.llm_response(prompt)
        return resp.content.strip()

# DecisionAgent: output hypotheses and scores
class DecisionAgent:
    def step(self, hypo_scores: dict[str, float]) -> str:
        lines = []
        for h_id, score in hypo_scores.items():
            text = hypo_lib.get(h_id)
            if text:
                lines.append(f"Hypothesis: {text}  Score: {score}")
        return "\n".join(lines)

# Extract overall score from CriticAgent feedback
def extract_overall_score(feedback: str) -> float:
    m = re.search(r"Overall Score:\s*(\d+\.?\d*)", feedback)
    return float(m.group(1)) if m else 0.0

# Full pipeline: hypothesis generation, evaluation, refinement
def run_full_pipeline(
    background: str,
    n_iterations: int = 1,
    max_lit_per_hypo: int = 8,
    direct_edge_limit: int = 30,
    node_limit: int = 30,
    pubmed_min_results: int = 1,
    pubmed_max_results: int = 4,
    record_dir: str = 'logs'
):
    import os, json
    RECORD_DIR = record_dir
    os.makedirs(RECORD_DIR, exist_ok=True)
    max_rounds = n_iterations
    MAX_LIT_PER_HYPO = max_lit_per_hypo

    initial_path = os.path.join(RECORD_DIR, "Initial_Hypothesis.jsonl")
    round_paths = {
        rnd: os.path.join(RECORD_DIR, f"Round{rnd}.jsonl")
        for rnd in range(1, max_rounds+1)
    }

    logs, hypo_scores, hypo_feedbacks, all_hypotheses_info = [], {}, {}, []
    hypothesis_lit_dict = {}
    bg_text = background

    kg_agent = KGAgent()
    pubmed_agent = PubmedAgent(min_results=pubmed_min_results, max_results=pubmed_max_results)
    critic = CriticAgent()
    revision = RevisionAgent(pubmed_min_results=pubmed_min_results, pubmed_max_results=pubmed_max_results)
    refine_agent = RefineAgent()

    domain = DomainSelectorAgent().step(background)
    write_agent_log("DomainSelectorAgent", background, domain)

    kg_context = DiseaseExplorerAgent().step(
        background, domain,
        direct_edge_limit=direct_edge_limit,
        node_limit=node_limit
    )

    write_agent_log("KGAgent", {"background": background, "domain": domain}, kg_context)

    sci_raw = ScientistAgent().step(background, kg_context)
    write_agent_log("ScientistAgent", {"background": background, "kg_context": kg_context}, sci_raw)

    initial_hypos = [h.strip() for h in sci_raw.split("\n") if h.strip()]
    initial_evids = [""] * len(initial_hypos)

    def _strip_header(bg: str) -> str:
        parts = bg.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else bg

    with open(initial_path, "a", encoding="utf-8") as outf:
        bg_clean = _strip_header(background)
        for hypo in initial_hypos:
            outf.write(json.dumps({
                "background": bg_clean,
                "hypothesis": hypo
            }, ensure_ascii=False) + "\n")

    all_initial_ids = []
    for idx, (hypo, ev) in enumerate(zip(initial_hypos, initial_evids)):
        h_id = idx
        all_initial_ids.append(h_id)
        ev_full = ev if ev.startswith("EVIDENCE:") else ev

        pub_json = pubmed_agent.step(
            [hypo],
            hypothesis=hypo,
            feedback="",
            min_results=pubmed_min_results,
            max_results=pubmed_max_results
        )

        write_agent_log("PubmedAgent", {"hypothesis": hypo, "domain": domain}, pub_json)

        pubmed_infos = extract_pubmed_info(pub_json, max_n=MAX_LIT_PER_HYPO)
        hypothesis_lit_dict[h_id] = pubmed_infos

        lit_txt = "\n".join(
            f"PMID: {x.get('pmid', '')}\nTitle: {x.get('title', '')}\nAbstract: {x.get('abstract', '')}"
            for x in pubmed_infos
        )
        fb = critic.step(bg_text, lit_txt, hypo)
        write_agent_log("CriticAgent", {"background": bg_text, "literature": lit_txt, "hypothesis": hypo}, fb)

        score = extract_overall_score(fb)
        hypo_feedbacks[h_id] = fb
        hypo_scores[h_id] = score

        all_hypotheses_info.append({
            "from_hypo": idx,
            "type": "Scientist",
            "hypothesis": hypo,
            "evidence": ev_full,
            "score": score
        })

    for idx, h_id in enumerate(all_initial_ids):
        curr_id = h_id
        curr_txt = initial_hypos[idx]
        prior_evidence = ""
        for rnd in range(1, max_rounds + 1):
            fb = hypo_feedbacks[curr_id]
            actions, info, metrics_texts, *_ = revision.step(
                fb, curr_txt, domain, background,
                direct_edge_limit=direct_edge_limit,
                node_limit=node_limit,
            )
            write_agent_log("RevisionAgent", {"feedback": fb, "hypothesis": curr_txt, "domain": domain, "background": background},
                {"actions": actions, "info": info, "metrics_texts": metrics_texts})

            curr_lits = hypothesis_lit_dict.get(curr_id, []).copy()
            for lbl, cnt in info:
                if lbl == "pubmed":
                    new_infos = extract_pubmed_info(cnt, max_n=MAX_LIT_PER_HYPO)
                    existing = {x.get("pmid") for x in curr_lits}
                    for x in new_infos:
                        if x.get("pmid") not in existing and len(curr_lits) < MAX_LIT_PER_HYPO:
                            curr_lits.append(x)
            ref_raw = refine_agent.step(curr_txt, info, metrics_texts, prior_evidence)
            write_agent_log("RefineAgent", {"hypothesis": curr_txt, "info": info, "metrics_texts": metrics_texts}, ref_raw)
            ref_lines = [line.strip() for line in ref_raw.splitlines() if line.strip()]
            ref_txt = ref_lines[-1] if ref_lines else ""
            if not ref_txt:
                break
            lit_txt2 = "\n".join(
                f"PMID: {x.get('pmid', '')}\nTitle: {x.get('title', '')}\nAbstract: {x.get('abstract', '')}"
                for x in curr_lits[:MAX_LIT_PER_HYPO]
            )
            fb2 = critic.step(bg_text, lit_txt2, ref_txt)
            write_agent_log("CriticAgent", {"background": bg_text, "literature": lit_txt2, "hypothesis": ref_txt}, fb2)
            score2 = extract_overall_score(fb2)
            hypo_feedbacks[curr_id] = fb2
            hypo_scores[curr_id] = score2
            all_hypotheses_info.append({
                "from_hypo": idx,
                "type": "Refined",
                "hypothesis": ref_txt,
                "evidence": "",
                "score": score2
            })
            with open(round_paths[rnd], "a", encoding="utf-8") as outf:
                bg_clean = _strip_header(background)
                outf.write(json.dumps({
                    "background": bg_clean,
                    "hypothesis": ref_txt
                }, ensure_ascii=False) + "\n")
            prior_evidence = ""
            curr_id, curr_txt = h_id, ref_txt

    return all_hypotheses_info
