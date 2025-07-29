import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any
import json
import time
import re
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from .llm_config import gpt4o_mini_config_graph


def extract_json_from_response(text):
    """从LLM输出中自动提取纯JSON对象"""
    if text is None:
        raise ValueError("No LLM output to extract JSON from.")
    text = text.strip()
    if text.startswith("```"):
        # 去掉markdown代码块包裹
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    # 查找首个{...}
    matches = re.findall(r'\{[\s\S]*\}', text)
    if matches:
        return json.loads(matches[0])
    # fallback: 尝试直接parse
    return json.loads(text)

def clean_llm_config(llm_config):
    # Clean and standardize LLM config
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

class KeywordQueryAgent(ChatAgent):
    # PubMed keyword query strategy agent
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4o_mini_config_graph)),
            system_message=(
                "You are an expert biomedical PubMed search strategist.\n"
                "Task: Given a list of biomedical keywords, organize them into synonym groups (OR within each),\n"
                "choose whether to combine groups with AND or OR, and propose multiple PubMed queries.\n"
                "Important:\n"
                "- NEVER split biomedical entity names (e.g., genes, proteins, chemicals) into characters or numbers. For example, always treat 'GPR153', 'TP53', or 'VEGFR2' as single, indivisible keywords. Do not group or split them into ['G','P','R','153'] or similar.\n"
                "- Only group full terms, abbreviations, or full names as single units, not character-by-character or digit-by-digit.\n"
                "Output exactly one valid JSON with these fields:\n"
                "  \"groups\": list of lists of keywords, e.g. [[\"term1\",\"term1 synonym\"], ...]\n"
                "  \"group_logic\": \"AND\" or \"OR\"\n"
                "  \"multi_queries\": list of objects, each with:\n"
                "      \"groups\": list of lists, as above\n"
                "      \"group_logic\": \"AND\" or \"OR\"\n"
                "      \"notes\": concise rationale\n"
                "  \"notes\": overall concise guidance (e.g. date range, field tags, focus)\n"
                "Do NOT output anything else."
            )
        )
        cfg.name = "PubMed_KeywordQueryAgent"
        super().__init__(cfg)

    def get_strategy(self, keywords: str) -> Dict[str, Any]:
        # Get query strategy from LLM
        prompt = f"{self.config.system_message}\n\nKEYWORDS:\n{keywords}\n\nQUERY_STRATEGY:"
        resp = self.llm_response(prompt)
        # print("[DEBUG] LLM output:", resp)
        # print("[DEBUG] LLM output content:", getattr(resp, "content", None))
        if not resp or not getattr(resp, "content", None):
            raise RuntimeError("LLM did not return a response.")
        strat = extract_json_from_response(getattr(resp, "content", ""))

        mqs = strat.get("multi_queries", [])
        strat["multi_queries"] = [mq for mq in mqs if mq.get("group_logic","AND").upper()=="AND"]
        return strat

class HypothesisQueryAgent(ChatAgent):
    # PubMed query agent for hypothesis with feedback
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4o_mini_config_graph)),
            system_message=(
                "You are an expert biomedical PubMed search strategist.\n"
                "Task: Given a research hypothesis, its low-score feedback, and related keywords,\n"
                "identify under-supported concepts, group synonyms (OR within each), choose AND/OR logic, and propose multiple PubMed queries.\n"
                "Important:\n"
                "- NEVER split biomedical entity names (e.g., genes, proteins, chemicals) into characters or numbers. For example, always treat 'GPR153', 'TP53', or 'VEGFR2' as single, indivisible keywords. Do not group or split them into ['G','P','R','153'] or similar.\n"
                "- Only group full terms, abbreviations, or full names as single units, not character-by-character or digit-by-digit.\n"
                "Output exactly one valid JSON with these fields:\n"
                "  \"groups\": list of lists of keywords\n"
                "  \"group_logic\": \"AND\" or \"OR\"\n"
                "  \"multi_queries\": list of objects, each with:\n"
                "      \"groups\": list of lists\n"
                "      \"group_logic\": \"AND\" or \"OR\"\n"
                "      \"notes\": concise rationale\n"
                "  \"notes\": overall concise guidance\n"
                "Do NOT output anything else."
            )
        )
        cfg.name = "PubMed_HypothesisQueryAgent"
        super().__init__(cfg)

    def get_strategy(self, hypothesis_and_feedback: str) -> Dict[str, Any]:
        # Get query strategy from LLM for hypothesis + feedback
        prompt = f"{self.config.system_message}\n\nINPUT:\n{hypothesis_and_feedback}\n\nQUERY_STRATEGY:"
        resp = self.llm_response(prompt)
        if not resp or not getattr(resp, "content", None):
            raise RuntimeError("LLM did not return a response.")
        strat = extract_json_from_response(getattr(resp, "content", ""))
        mqs = strat.get("multi_queries", [])
        strat["multi_queries"] = [mq for mq in mqs if mq.get("group_logic","AND").upper()=="AND"]
        return strat

def _safe_get(url: str, params: dict, max_retries: int = 3, backoff: float = 2.0):
    # Robust GET request with retries
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                return resp
            if attempt < max_retries - 1:
                time.sleep(backoff)
            else:
                resp.raise_for_status()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(backoff)
            else:
                raise

def build_pubmed_query(
    groups: List[Any],
    group_logic: str = "AND",
    field_pref: str = "MeSH/TIAB",
    start_date: str = "2018/01/01",
    end_date: str = None
) -> str:
    # Build PubMed query string from groups and logic
    if end_date is None:
        end_date = datetime.now().strftime("%Y/%m/%d")
    if isinstance(groups, dict):
        groups = list(groups.values())
    clauses = []
    for grp in groups:
        if isinstance(grp, dict) and "terms" in grp:
            terms = grp["terms"]
            joiner = grp.get("join", "OR").upper()
        elif isinstance(grp, list):
            terms = grp
            joiner = "OR"
        else:
            raise ValueError(f"Unsupported group format: {grp!r}")
        inner = []
        for term in terms:
            if "mesh" in field_pref.lower():
                inner.append(f"{term}[MeSH Terms]")
            if "tiab" in field_pref.lower():
                inner.append(f"{term}[TIAB]")
        clauses.append(f"({f' {joiner} '.join(inner)})")
    logic = group_logic.upper()
    query = f"{f' {logic} '.join(clauses)}"
    query += f' AND ("{start_date}"[PDAT] : "{end_date}"[PDAT])'
    query += ' AND "journal article"[pt]'
    return query

def pubmed_search(query: str, retmax: int = 20, api_key: str = None, sort: str = "relevance"):
    # PubMed search and fetch results using NCBI API
    search_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {'db':'pubmed','term':query,'retmax':retmax,'retmode':'json','sort':sort}
    key = api_key or os.getenv("PUBMED_API_KEY")
    if key: params['api_key'] = key
    resp = _safe_get(search_url, params)
    ids = resp.json().get('esearchresult',{}).get('idlist',[])
    if not ids: return []
    efetch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    efparams = {'db':'pubmed','id':','.join(ids),'retmode':'xml'}
    if key: efparams['api_key'] = key
    root = ET.fromstring(_safe_get(efetch_url, efparams).content)
    arts = []
    for art in root.findall('PubmedArticle'):
        med = art.find('MedlineCitation')
        art_el = med.find('Article')
        title = art_el.findtext('ArticleTitle') or ""
        abs_el = art_el.find('Abstract')
        abstract = " ".join(e.text for e in abs_el.findall('AbstractText') if e.text) if abs_el is not None else ""
        pub_date = med.findtext('DateCompleted/Year') or art_el.findtext('Journal/JournalIssue/PubDate/Year',"Unknown")
        pid = med.findtext('PMID') or "Unknown"
        arts.append({'id':pid,'title':title,'abstract':abstract,'pub_date':pub_date,'url':f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"})
    return arts

def adaptive_pubmed_search(
    strategy: Dict[str,Any],
    field_pref: str = "MeSH/TIAB",
    start_date: str = "2018/01/01",
    end_date: str = None,
    api_key: str = None,
    min_results: int = 3,
    max_results: int = 10,
    retmax_per_query: int = 20
):
    # Search PubMed adaptively for each query group, merge and deduplicate results
    if end_date is None:
        end_date = datetime.now().strftime("%Y/%m/%d")
    queries = strategy.get("multi_queries", [])
    if not queries:
        queries = [{
            "groups": strategy["groups"],
            "group_logic": strategy.get("group_logic", "AND"),
            "notes": strategy.get("notes", "")
        }]
    all_hits = []
    results_per_query = []
    for q in queries:
        try:
            q_groups = q["groups"]
            logic = q.get("group_logic", "AND")
            qstr = build_pubmed_query(q_groups, logic, field_pref, start_date, end_date)
            hits = pubmed_search(qstr, retmax=retmax_per_query, api_key=api_key)
        except Exception as ex:
            print(f"[WARN] query build failed for {q}: {ex}")
            continue
        if len(hits) < min_results and len(q_groups) > 1:
            for i, subgroup in enumerate(q_groups):
                try:
                    subqstr = build_pubmed_query([subgroup], "OR", field_pref, start_date, end_date)
                    subhits = pubmed_search(subqstr, retmax=retmax_per_query, api_key=api_key)
                    if subhits:
                        for h in subhits:
                            h["query_subgroup_idx"] = i
                        hits.extend(subhits)
                except Exception as ex:
                    print(f"[WARN] subgroup query failed: {ex}")
        unique = []
        seen = set()
        for h in hits:
            if h['id'] not in seen:
                unique.append(h)
                seen.add(h['id'])
        unique.sort(key=lambda a: int(a['pub_date'][:4]) if a['pub_date'][:4].isdigit() else 0, reverse=True)
        results_per_query.append(unique[:5])
    final_articles = []
    seen_ids = set()
    for lst in results_per_query:
        for h in lst:
            if h['id'] not in seen_ids:
                final_articles.append(h)
                seen_ids.add(h['id'])
    if len(final_articles) < min_results:
        status = "Too few articles after all attempts"
    else:
        status = f"Success: {len(final_articles)} unique articles"
    return final_articles[:max_results], strategy.get("groups", []), status
