from __future__ import annotations
import argparse
import json
from datetime import datetime
from typing import List, Sequence
import os
from dotenv import load_dotenv
load_dotenv()

from utils.llm_config import BACKGROUND_SUMMARISER_CONFIG
from utils.log_utils import write_agent_log
from agents_evidence import (
    ChatAgent, ChatAgentConfig,
    ensure_specific_llm_config, clean_llm_config,
    run_full_pipeline,                        
    DomainSelectorAgent, DiseaseExplorerAgent, ScientistAgent,
    call_pubmed_search,
    KeywordExtractorAgent
)


# ----------------------------- LLM agent -------------------------------- #
class BackgroundSummariserAgent(ChatAgent):
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(BACKGROUND_SUMMARISER_CONFIG)),
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

        # 兼容所有情况
        if resp is None:
            background = ""
        elif hasattr(resp, "content"):
            background = resp.content.strip()
        elif isinstance(resp, str):
            background = resp.strip()
        else:
            background = str(resp).strip()

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
    start_date: str = "2019/01/01",
    end_date: str = "2023/12/31",
    retmax: int = 10,
) -> str:
    if related_articles is None:
        related_articles = []
    blocks = [json.dumps(a, ensure_ascii=False) for a in related_articles]
    background = BackgroundSummariserAgent().summarise(disease, core_genes, blocks)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    header = (
        f"### Auto-generated background for {disease} (genes: {', '.join(core_genes)}), "
        f"PubMed window: {start_date} - {end_date} (generated {timestamp} UTC)\n\n"
    )
    return header + background



def run_background_only(
    disease: str,
    core_genes: Sequence[str],
    start_date: str = "2019/01/01",
    end_date: str = "2023/12/31",
    min_results: int = 3,
    max_results: int = 10,
):
    """No refinement loop; just BG + initial hypotheses."""
    articles_result = call_pubmed_search(
        keywords=[disease] + list(core_genes),
        min_results=min_results,
        max_results=max_results,
        start_date=start_date,
        end_date=end_date, 
    )
    articles = articles_result["articles"]

    background = build_background_summary(
        disease=disease,
        core_genes=core_genes,
        related_articles=articles,
        start_date=start_date,
        end_date=end_date,
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
    start_date: str = "2019/01/01",
    end_date: str = "2023/12/31",
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
        start_date=start_date,
        end_date=end_date, 
    )
    articles = art_res["articles"]

    background = build_background_summary(
        disease=disease,
        core_genes=core_genes,
        related_articles=articles,
        start_date=start_date,
        end_date=end_date, 
        retmax=max_results
    )

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

    params = dict(
        start_date="2019/01/01",
        end_date="2023/12/31",
        min_results=3,
        max_results=10,
        node_limit=50,
        direct_edge_limit=30,
        max_paths=0,
        n_iterations=3,
        max_articles_per_round=10,
    )

    # Set environment variables to override global defaults
    if 'disable_pubmed' in kwargs:
        os.environ['DISABLE_PUBMED'] = 'True' if kwargs['disable_pubmed'] else 'False'

    if 'disable_kg' in kwargs:
        os.environ['DISABLE_KG'] = 'True' if kwargs['disable_kg'] else 'False'

    for key in kwargs:
        if key not in ['disable_pubmed', 'disable_kg']:
            params[key] = kwargs[key]  

    item = auto_structure_item(topic)
    disease = item.get("disease", "") # type: ignore
    core_genes = item.get("core_genes", []) # type: ignore

    res = run_biodisco_full(disease, core_genes, **params)# type: ignore

    return res['all_hypotheses'][-1]['hypothesis']
