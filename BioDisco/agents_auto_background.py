from __future__ import annotations
import argparse
import json
from datetime import datetime
from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from utils.llm_config import BACKGROUND_SUMMARISER_CONFIG
from utils.log_utils import write_agent_log
from agents_evidence import (
    ChatAgent, ChatAgentConfig,
    ensure_specific_llm_config, clean_llm_config,
    run_full_pipeline,                        
    DomainSelectorAgent, DiseaseExplorerAgent, ScientistAgent,
    call_pubmed_search
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


# ------------------------------- CLI ------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BioDisco pipelines")
    p.add_argument("--in",  dest="input_jsonl",  default="test.jsonl")
    p.add_argument("--out", dest="output_jsonl", default="result.jsonl")
    p.add_argument("--mode", choices=["simple", "full"], default="full")

    # PubMed / BG
    p.add_argument("--start_date", type=str, default="2019/01/01")
    p.add_argument("--end_date",   type=str, default="2023/12/31")

    p.add_argument("--min_results", type=int, default=3)
    p.add_argument("--max_results", type=int, default=10)

    p.add_argument("--node_limit", type=int, default=50)
    p.add_argument("--direct_edge_limit", type=int, default=30)
    p.add_argument("--max_paths",      type=int, default=0)
    p.add_argument("--n_iterations",   type=int, default=3)
    p.add_argument("--max_articles_per_round", type=int, default=10)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline_on_file(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        mode="full" if args.mode == "full" else "simple",
        start_data=args.start_data,
        end_data=args.end_data, 
        min_results=args.min_results,
        max_results=args.max_results,
        node_limit=args.node_limit,
        direct_edge_limit=args.direct_edge_limit,
        max_paths=args.max_paths,
        n_iterations=args.n_iterations,
        max_articles_per_round=args.max_articles_per_round,
    )

