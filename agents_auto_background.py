from __future__ import annotations
import os
import sys
sys.path.insert(0, r"D:\DFKI\SciAgentsDiscovery-openai\SciAgentsDiscovery-main")
import re
import json
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from datetime import datetime
from typing import List, Sequence

# Import main components and agent classes
from ScienceDiscovery.log_utils import write_agent_log
from ScienceDiscovery.agents_evidence import (
    KeywordExtractorAgent,
    ChatAgent, ChatAgentConfig,
    gpt4turbo_mini_config,
    ensure_specific_llm_config,
    clean_llm_config,
    run_full_pipeline,
    DomainSelectorAgent,
    DiseaseExplorerAgent,
    ScientistAgent,
    call_pubmed_search,
    DISABLE_PUBMED
)



# LLM agent for summarizing PubMed articles into short background
class BackgroundSummariserAgent(ChatAgent):
    def __init__(self):
        cfg = ChatAgentConfig(
            llm=ensure_specific_llm_config(clean_llm_config(gpt4turbo_mini_config)),
            system_message=(
                "BackgroundSummariserAgent:\n"
                "Task: You are given a list of PubMed article metadata blocks about a specific disease and a set of core genes or biological entities.\n"
                "Write a concise, well-structured background paragraph (less than 150 words) that summarizes key mechanistic insights and highlights the relationships between the core genes and disease-relevant biological processes (such as EMT, inflammation, senescence, signaling, etc.).\n"
                "Requirements:\n"
                "- Clearly explain how the core genes are linked to disease mechanisms, pathways, or phenotypes based on the literature.\n"
                "- Emphasize causal or regulatory connections when possible, rather than just listing associations.\n"
                "- Do not copy sentences verbatim from abstracts. Always synthesize and paraphrase information in your own words.\n"
                "- Use clear, logical, and scientifically precise language.\n"
                "- Avoid including superfluous or generic information; focus on mechanistic insights most relevant to the disease and core genes.\n"
                "- Stay within 150 words total."
                "If there is little or no literature evidence available, be extremely cautious: **do not fabricate mechanistic details or connections**. Instead, limit the background to well-known and basic disease or gene facts that are widely accepted, or explicitly state that no strong mechanistic insights can be summarized due to insufficient evidence.\n"
                "When in doubt, err on the side of brevity and avoid making unsupported or speculative claims.\n"
            )
        )
        cfg.name = "BackgroundSummariserAgent"
        super().__init__(cfg)

    # Summarize PubMed articles for given disease and genes
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
            background
        )
        return background


# Build background summary for a disease and gene list (main pipeline)
def build_background_summary(
    disease: str,
    core_genes: Sequence[str],
    related_articles: list = None,   
    start_year: int = 2019,
    retmax: int = 10,
) -> str:
    if related_articles is None:
        related_articles = []
    blocks = [json.dumps(a, ensure_ascii=False) for a in related_articles]
    background = BackgroundSummariserAgent().summarise(disease, core_genes, blocks)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    header = (
        f"### Auto-generated background for {disease} (genes: {', '.join(core_genes)}), "
        f"PubMed cut-off ≥{start_year} (generated {timestamp} UTC)\n\n"
    )
    return header + background


# # Ablation - no iteration
# if __name__ == "__main__":
#     input_jsonl = r"D:\DFKI\SciAgentsDiscovery-openai\test.jsonl"

#     output_jsonl = r"D:\DFKI\SciAgentsDiscovery-openai\SciAgentsDiscovery-main\ScienceDiscovery\test_1.jsonl"

#     with open(input_jsonl, "r", encoding="utf-8") as fin, \
#          open(output_jsonl, "w", encoding="utf-8") as fout:

#         for idx, line in enumerate(fin, 1):
#             try:
#                 item = json.loads(line)
#                 disease = item.get("disease", "")
#                 core_genes = item.get("core_genes", [])

#                 articles_result = call_pubmed_search(
#                     keywords=[disease] + list(core_genes),
#                     min_results=3,
#                     max_results=10
#                 )
#                 articles = articles_result["articles"]

#                 background = build_background_summary(
#                     disease=disease,
#                     core_genes=core_genes,
#                     related_articles=articles,   
#                     start_year=2019,
#                     retmax=10,
#                 )

#                 # KG context
#                 domain_agent = DomainSelectorAgent()
#                 domain = domain_agent.step(background)
#                 kg_context = DiseaseExplorerAgent().step(background, domain)
                
#                 # initial hypothesis
#                 sci_agent = ScientistAgent()
#                 sci_raw = sci_agent.step(background, kg_context)
#                 hypotheses = [h.strip() for h in sci_raw.split("\n") if h.strip()]

#                 out_item = {
#                     "disease": disease,
#                     "core_genes": core_genes,
#                     "hypotheses": hypotheses
#                 }
#                 fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
#                 print(f"===> Done: {disease}, {len(hypotheses)} initial hypotheses written.")
#             except Exception as e:
#                 print(f"[ERROR] {e}")
#                 continue



# BioDisco full-pipiline
if __name__ == "__main__":
    input_jsonl = r"D:\DFKI\SciAgentsDiscovery-openai\test.jsonl"

    output_jsonl = r"D:\DFKI\SciAgentsDiscovery-openai\SciAgentsDiscovery-main\ScienceDiscovery\test_1.jsonl"

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, 1):
            try:
                item = json.loads(line)
                disease = item.get("disease", "")
                core_genes = item.get("core_genes", [])
                # Step 1: Generate background summary
                articles_result = call_pubmed_search(
                    keywords=[disease] + list(core_genes),
                    min_results=3,
                    max_results=10
                )
                articles = articles_result["articles"]

                background = build_background_summary(
                    disease=disease,
                    core_genes=core_genes,
                    related_articles=articles,  
                    start_year=2019,
                    retmax=10,
                )
                # Step 2: Run full pipeline (KG, literature, revision, refine)
                all_hypotheses_info = run_full_pipeline(background)
                # Step 3: Write all hypotheses and scores
                out_item = {
                    "disease": disease,
                    "core_genes": core_genes,
                    "all_hypotheses": [
                        {
                            "hypothesis": h.get("hypothesis", ""),
                            "score": h.get("score", None),
                            "from_hypo": h.get("from_hypo", None),
                            "type": h.get("type", ""),
                            "evidence": h.get("evidence", "")
                        }
                        for h in all_hypotheses_info
                    ]
                }
                fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                print(f"===> Done: {disease}, {len(all_hypotheses_info)} hypotheses written.")
            except Exception as e:
                print(f"[ERROR] {e}")
                continue





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

