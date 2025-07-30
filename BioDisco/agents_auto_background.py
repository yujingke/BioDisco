from __future__ import annotations
import argparse
import json
from datetime import datetime
from typing import List, Sequence
import os

from dotenv import load_dotenv
load_dotenv()

from .utils.log_utils import write_agent_log
from .agents_evidence import (
    ChatAgent, ChatAgentConfig,
    gpt4turbo_mini_config, ensure_specific_llm_config, clean_llm_config,
    run_full_pipeline,                        # <-- 你已有的函数
    DomainSelectorAgent, DiseaseExplorerAgent, ScientistAgent, KeywordExtractorAgent,
    call_pubmed_search
)

# ----------------------------- LLM agent -------------------------------- #
class BackgroundSummariserAgent(ChatAgent):
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=(
                "BackgroundSummariserAgent:\n"
                "Task: Given PubMed article metadata about a disease and core genes, write a concise "
                "background paragraph (<150 words) highlighting mechanistic links.\n"
                "Requirements:\n"
                "- Explain how core genes connect to disease mechanisms/pathways/phenotypes.\n"
                "- Prefer causal/regulatory relations over loose associations.\n"
                "- No verbatim copy; paraphrase concisely and precisely.\n"
                "- Avoid generic fluff; focus on mechanistic insights.\n"
                "- If evidence is scarce, do NOT fabricate; state limitation briefly.\n"
            )
        )
        cfg.name = "BackgroundSummariserAgent"
        super().__init__(cfg)

    def summarise(self, disease: str, genes: Sequence[str], article_blocks: List[str]) -> str:
        joined = "\n".join(article_blocks[:20])
        prompt = (
            f"{self.config.system_message}\n"
            f"Disease: {disease}\n"
            f"Genes: {', '.join(genes)}\n"
            f"Articles (JSON blocks):\n{joined}\n\n"
            "Produce concise background (<=150 words)"
        )
        resp = self.llm_response(prompt)
        background = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        write_agent_log(
            "BackgroundSummariserAgent",
            {"disease": disease, "genes": genes, "articles": article_blocks},
            background,
        )
        return background


# -------------------------- Helper functions ---------------------------- #
def build_background_summary(
    disease: str,
    core_genes: Sequence[str],
    related_articles: list | None = None,
    start_year: int = 2019,
    retmax: int = 10,
) -> str:
    if related_articles is None:
        related_articles = []
    blocks = [json.dumps(a, ensure_ascii=False) for a in related_articles]
    background = BackgroundSummariserAgent().summarise(disease, core_genes, blocks)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    header = (
        f"### Auto-generated background for {disease} (genes: {', '.join(core_genes)}), "
        f"PubMed cut-off ≥{start_year} (generated {timestamp} UTC)\n\n"
    )
    return header + background


def run_background_only(
    disease: str,
    core_genes: Sequence[str],
    start_year: int = 2019,
    min_results: int = 3,
    max_results: int = 10,
):
    """No refinement loop; just BG + initial hypotheses."""
    articles_result = call_pubmed_search(
        keywords=[disease] + list(core_genes),
        min_results=min_results,
        max_results=max_results,
    )
    articles = articles_result["articles"]

    background = build_background_summary(
        disease=disease,
        core_genes=core_genes,
        related_articles=articles,
        start_year=start_year,
        retmax=max_results,
    )

    domain = DomainSelectorAgent().step(background)
    kg_context = DiseaseExplorerAgent().step(background, domain)

    sci_raw = ScientistAgent().step(background, kg_context)
    hypotheses = [h.strip() for h in sci_raw.split("\n") if h.strip()]

    return {
        "disease": disease,
        "core_genes": core_genes,
        "background": background,
        "articles": articles,
        "hypotheses": hypotheses,
    }


def run_biodisco_full(
    disease: str,
    core_genes: Sequence[str],
    start_year: int = 2019,
    min_results: int = 3,
    max_results: int = 10,
    node_limit: int = 50,
    direct_edge_limit: int = 30,
    max_paths: int = 0,
    n_iterations: int = 3,
    max_articles_per_round: int = 10,
):
    """Full pipeline; pass all args to run_full_pipeline."""
    articles = []

    art_res = call_pubmed_search(
        keywords=[disease] + list(core_genes),
        min_results=min_results,
        max_results=max_results,
    )
    articles = art_res["articles"]

    background = build_background_summary(
        disease=disease,
        core_genes=core_genes,
        related_articles=articles,
        start_year=start_year,
        retmax=max_results
    )

    # 传递参数到 run_full_pipeline
    all_hypotheses_info = run_full_pipeline(
        background,
        n_iterations=n_iterations,
        max_lit_per_hypo=max_articles_per_round
    )

    return {
        "disease": disease,
        "core_genes": core_genes,
        "background": background,
        "articles": articles,
        "all_hypotheses": [
            {
                "hypothesis": h.get("hypothesis", ""),
                "score": h.get("score", None),
                "from_hypo": h.get("from_hypo", None),
                "type": h.get("type", ""),
                "evidence": h.get("evidence", ""),
            }
            for h in all_hypotheses_info
        ],
    }


def run_pipeline_on_file(
    input_jsonl: str,
    output_jsonl: str,
    mode: str = "full",
    **kwargs,
):
    runner = run_biodisco_full if mode == "full" else run_background_only

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, 1):
            try:
                item = json.loads(line)
                disease = item.get("disease", "")
                core_genes = item.get("core_genes", [])
                result = runner(disease, core_genes, **kwargs)
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"[{mode.upper()}] {idx}: {disease} ✓")
            except Exception as e:
                print(f"[ERROR] line {idx}: {e}")
                continue

def is_structured_entry(item):
    return "disease" in item and "core_genes" in item

def auto_structure_item(item):
    if is_structured_entry(item):
        return item
    if "text" in item:
        input_text = item["text"]
    elif isinstance(item, str):
        input_text = item
    else:
        input_text = next(iter(item.values()))
    kws = KeywordExtractorAgent().extract(input_text)
    disease = ""
    core_genes = []
    for kw in kws:
        if not disease:
            disease = kw
        else:
            core_genes.append(kw)

    if not core_genes and disease:
        core_genes = [disease]
    return {"disease": disease, "core_genes": core_genes}

def generate(topic: str, **kwargs) -> str:
    """
    Generate a structured item from a topic string.
    
    Args:
        topic (str): Input topic string.
        
    Returns:
        Dict: Structured item with disease and core genes.
    """
    params = dict(
        start_year=2019,
        min_results=3,
        max_results=10,
        n_iterations=3,
        max_articles_per_round=10,
    )

    # Set environment variables to override global defaults
    if 'disable_pubmed' in kwargs:
        if kwargs['disable_pubmed']:
            os.environ['DISABLE_PUBMED'] = 'True'
        else:
            os.environ['DISABLE_PUBMED'] = 'False'

    if 'disable_kg' in kwargs:
        if kwargs['disable_kg']:
            os.environ['DISABLE_KG'] = 'True'
        else:
            os.environ['DISABLE_KG'] = 'False'
    
    # Update params with any additional kwargs
    for key in kwargs:
        if key in params:
            params[key] = kwargs[key]
        elif key not in ['disable_pubmed', 'disable_kg']:
            print(f"Warning: Unrecognized parameter '{key}' in kwargs. It will be ignored.")

    item = auto_structure_item(topic)
    disease = item.get("disease", "")
    core_genes = item.get("core_genes", [])
    
    res = run_biodisco_full(disease, core_genes, **params)

    return res['all_hypotheses'][-1]['hypothesis']
