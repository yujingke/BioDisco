from __future__ import annotations
import argparse
import json
from datetime import datetime
from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from utils.log_utils import write_agent_log
from agents_evidence import (
    ChatAgent, ChatAgentConfig,
    gpt4turbo_mini_config, ensure_specific_llm_config, clean_llm_config,
    run_full_pipeline,                        # <-- 你已有的函数
    DomainSelectorAgent, DiseaseExplorerAgent, ScientistAgent,
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


# ------------------------------- CLI ------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BioDisco pipelines")
    p.add_argument("--in",  dest="input_jsonl",  default="test.jsonl")
    p.add_argument("--out", dest="output_jsonl", default="result.jsonl")
    p.add_argument("--mode", choices=["simple", "full"], default="full")

    # PubMed / BG
    p.add_argument("--start_year", type=int, default=2019)
    p.add_argument("--min_results", type=int, default=3)
    p.add_argument("--max_results", type=int, default=10)

    # run_full_pipeline 常用参数
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
        start_year=args.start_year,
        min_results=args.min_results,
        max_results=args.max_results,
        node_limit=args.node_limit,
        direct_edge_limit=args.direct_edge_limit,
        max_paths=args.max_paths,
        n_iterations=args.n_iterations,
        max_articles_per_round=args.max_articles_per_round,
    )

# Temporal eval gpt3.5
# import json

# def clean_unicode_garbage(text):
#     text = text.replace("â\x80\x93", "-").replace("â\x80\x94", "-")
#     text = text.replace("â€“", "-").replace("â€”", "-")
#     return text

# def clean_text(text):
#     return (text.replace("Ã¢ÂÂ", "–")
#                 .replace("Ã¢ÂÂ", "—")
#                 .replace("â€™", "'")
#                 .replace("â€“", "–")
#                 .replace("â€”", "—")
#                 .replace("ÃÂº", "κ")
#                 .replace("â€œ", '"')
#                 .replace("â€�", '"')
#                 .replace("Ã¢ÂÂ", "'")
#                 .replace("Ã¢ÂÂ", "'")
#                 .replace("Ã¢ÂÂ", '"')
#                 .replace("Ã¢ÂÂ", '"')
#                 .replace("ÃÂ±", "±")
#                 .replace("ÃÂµ", "µ")
#                 .replace("ÃÂ°C", "°C")
#                 .replace("ÃÂ", "")
#             )

# def get_scientist_initial_hypo(background):

#     try:
#         from agents_background import ScientistAgent, DiseaseExplorerAgent, DomainSelectorAgent
#         domain = DomainSelectorAgent().step(background)
#         kg_context = DiseaseExplorerAgent().step(background, domain)
#         sci_raw = ScientistAgent().step(background, kg_context)
#         hypos = [h.strip() for h in sci_raw.split("\n") if h.strip()]
#         return hypos[0] if hypos else "Failed to generate hypothesis"
#     except Exception as e:
#         print(f"[ERROR][ScientistAgent fallback] {e}")
#         return "Failed to generate hypothesis"

# if __name__ == "__main__":
#     bg_input_jsonl = r"D:\DFKI\SciAgentsDiscovery-openai\test.jsonl"
#     out_jsonl = r"D:\DFKI\SciAgentsDiscovery-openai\result.jsonl"

#     with open(bg_input_jsonl, "r", encoding="utf-8") as for_read, \
#          open(out_jsonl, "a", encoding="utf-8") as for_write:

#         for idx, line in enumerate(for_read, 1):
#             try:
#                 line = line.strip()
#                 if not line or not line.lstrip().startswith("{"):
#                     continue
#                 line = clean_unicode_garbage(line)
#                 try:
#                     obj = json.loads(line)
#                 except Exception as e:
#                     print(f"[ERROR] JSON decode failed: {e} (line: {repr(line)})")
#                     continue

#                 background = clean_text(obj.get("background", "").strip())
#                 if not background:
#                     continue

#                 print(f"\n==== [{idx}] Running full pipeline ====")
#                 try:
#                     all_hypotheses_info = run_full_pipeline(background)
#                     refineds = [h for h in all_hypotheses_info if h.get('type') == 'Refined']
#                     initials = [h for h in all_hypotheses_info if h.get('type') == 'Scientist']
#                     if refineds:
#                         best = max(refineds, key=lambda x: x.get('score', float('-inf')))
#                         final_hypo = best.get('hypothesis', '').strip()
#                     elif initials:
#                         best = max(initials, key=lambda x: x.get('score', float('-inf')))
#                         final_hypo = best.get('hypothesis', '').strip()
#                     else:
#                         final_hypo = ""
                     
#                     if not final_hypo:
#                         print("[WARN] No refined or initial hypothesis, using fallback scientist only.")
#                         final_hypo = get_scientist_initial_hypo(background)
#                 except Exception as e:
#                     print(f"[ERROR][Pipeline] {e}\n[INFO] Fallback: Running ScientistAgent only for idx={idx}")
#                     final_hypo = get_scientist_initial_hypo(background)

#                 if not final_hypo:
#                     final_hypo = "No hypothesis generated."

#                 out_item = {
#                     "background": background,
#                     "final_hypothesis": clean_text(final_hypo)
#                 }
#                 for_write.write(json.dumps(out_item, ensure_ascii=False) + "\n")
#                 for_write.flush()  
#                 print(f"===> Done: final hypothesis: {final_hypo}")

#             except Exception as e:
#                 print(f"[ERROR] {e} (line content: {repr(line)})")

#                 try:
#                     background = clean_text(obj.get("background", "").strip())
#                 except:
#                     background = ""
#                 try:
#                     fallback_hypo = clean_text(obj.get("hypothesis", "").strip())
#                 except:
#                     fallback_hypo = "No hypothesis generated."
#                 if background and fallback_hypo:
#                     out_item = {
#                         "background": background,
#                         "final_hypothesis": fallback_hypo
#                     }
#                     for_write.write(json.dumps(out_item, ensure_ascii=False) + "\n")
#                     for_write.flush()
#                     print(f"[ERROR][Fallback] Wrote original hypothesis for idx={idx}")
#                 continue

