#!/usr/bin/env python3
"""
Basic Usage Example for BioDisco Package

This example demonstrates the core functionality of BioDisco including:
- Creating and managing hypotheses
- Literature management with PubMed integration
- Knowledge graph operations
- Evidence linking

Author: BioDisco Team
"""

import os
from BioDisco import (
    HypothesisLibrary,
    LiteratureLibrary, 
    KGLibrary,
    EvidenceLibrary,
    write_agent_log
)

def main():
    """Main demonstration function."""
    print("🧬 BioDisco Basic Usage Example 🧬")
    print("=" * 50)
    
    # Initialize libraries
    print("\n1. Initializing BioDisco Libraries...")
    hypo_lib = HypothesisLibrary()
    lit_lib = LiteratureLibrary()
    kg_lib = KGLibrary()
    evidence_lib = EvidenceLibrary()
    print("✅ All libraries initialized successfully!")
    
    # Working with hypotheses
    print("\n2. Working with Hypotheses...")
    hypothesis_text = "Vitamin D deficiency increases COVID-19 severity and mortality risk"
    hypothesis_id = hypo_lib.add(hypothesis_text)
    print(f"✅ Added hypothesis: {hypothesis_text}")
    print(f"   Hypothesis ID: {hypothesis_id}")
    
    # Test deduplication
    duplicate_id = hypo_lib.add(hypothesis_text)
    print(f"✅ Deduplication test: {hypothesis_id == duplicate_id}")
    
    # Working with literature
    print("\n3. Working with Literature...")
    paper_data = {
        "pmid": "33234567",
        "title": "COVID-19 and Vitamin D: A Systematic Review and Meta-Analysis",
        "abstract": "This systematic review and meta-analysis examines the relationship between vitamin D levels and COVID-19 outcomes. We found significant associations between vitamin D deficiency and increased severity of COVID-19 symptoms.",
        "authors": ["Smith, J.A.", "Johnson, B.C.", "Williams, D.E."],
        "journal": "Nature Medicine",
        "year": 2021,
        "doi": "10.1038/s41591-021-01234-5"
    }
    
    literature_id = lit_lib.add(paper_data)
    print(f"✅ Added literature: {paper_data['title']}")
    print(f"   Literature ID: {literature_id}")
    
    # Retrieve literature
    retrieved_paper = lit_lib.get(literature_id)
    print(f"✅ Retrieved paper PMID: {retrieved_paper['pmid']}")
    
    # Working with knowledge graph
    print("\n4. Working with Knowledge Graph...")
    
    # Add gene node
    vdr_gene = {
        "entity_id": "VDR_gene", 
        "name": "Vitamin D Receptor",
        "type": "Gene",
        "synonyms": ["VDR", "NR1I1"],
        "description": "Nuclear receptor for vitamin D3"
    }
    gene_id = kg_lib.add_node(vdr_gene)
    print(f"✅ Added gene node: {vdr_gene['name']}")
    
    # Add disease node
    covid19_disease = {
        "entity_id": "COVID19_disease",
        "name": "COVID-19", 
        "type": "Disease",
        "synonyms": ["SARS-CoV-2 infection", "Coronavirus Disease 2019"],
        "description": "Infectious disease caused by SARS-CoV-2"
    }
    disease_id = kg_lib.add_node(covid19_disease)
    print(f"✅ Added disease node: {covid19_disease['name']}")
    
    # Add vitamin D node
    vitamin_d = {
        "entity_id": "VitaminD_compound",
        "name": "Vitamin D",
        "type": "Compound",
        "synonyms": ["Cholecalciferol", "25-hydroxyvitamin D"],
        "description": "Fat-soluble vitamin important for calcium absorption"
    }
    compound_id = kg_lib.add_node(vitamin_d)
    print(f"✅ Added compound node: {vitamin_d['name']}")
    
    # Add relationships
    gene_disease_edge = {
        "source": "VDR_gene",
        "target": "COVID19_disease",
        "relation": "associated_with",
        "evidence_strength": 0.75,
        "source_type": "literature_mining"
    }
    edge1_id = kg_lib.add_edge(gene_disease_edge)
    print(f"✅ Added edge: VDR gene ↔ COVID-19")
    
    compound_gene_edge = {
        "source": "VitaminD_compound", 
        "target": "VDR_gene",
        "relation": "binds_to",
        "evidence_strength": 0.95,
        "source_type": "experimental"
    }
    edge2_id = kg_lib.add_edge(compound_gene_edge)
    print(f"✅ Added edge: Vitamin D → VDR gene")
    
    # Linking evidence
    print("\n5. Linking Evidence...")
    evidence_id = evidence_lib.add(
        hypothesis_id=hypothesis_id,
        literature_ids=[literature_id],
        kg_node_ids=[gene_id, disease_id, compound_id],
        kg_edge_ids=[edge1_id, edge2_id],
        prev_hypothesis_ids=[]
    )
    print(f"✅ Created evidence link: {evidence_id}")
    
    # Display summary
    print("\n6. Summary...")
    print(f"   📚 Total hypotheses: {len(hypo_lib.all())}")
    print(f"   📖 Total literature: {len(lit_lib.all())}")
    print(f"   🔗 Total KG nodes: {len(kg_lib.all_nodes())}")
    print(f"   🔗 Total KG edges: {len(kg_lib.all_edges())}")
    print(f"   🧪 Total evidence links: {len(evidence_lib.all())}")
    
    # Log the example run
    write_agent_log(
        agent="BioDisco_Example",
        input_data={"example_type": "basic_usage"},
        output_data={
            "hypotheses_created": 1,
            "literature_added": 1,
            "kg_nodes_created": 3,
            "kg_edges_created": 2,
            "evidence_links": 1
        }
    )
    print("\n✅ Example completed successfully!")
    print("📋 Run log saved to run_log.jsonl")

def advanced_example():
    """Advanced example showing more complex operations."""
    print("\n🔬 Advanced BioDisco Features")
    print("=" * 50)
    
    # Initialize libraries
    hypo_lib = HypothesisLibrary()
    kg_lib = KGLibrary()
    
    # Multiple related hypotheses
    hypotheses = [
        "Vitamin D deficiency impairs immune system function",
        "Immune system dysfunction increases viral infection susceptibility", 
        "SARS-CoV-2 exploits compromised immune responses",
        "Vitamin D supplementation may reduce COVID-19 risk"
    ]
    
    hypothesis_ids = []
    for h in hypotheses:
        hid = hypo_lib.add(h)
        hypothesis_ids.append(hid)
        print(f"✅ Added: {h}")
    
    # Complex knowledge graph
    entities = [
        {"entity_id": "IL6", "name": "Interleukin-6", "type": "Protein"},
        {"entity_id": "TNF", "name": "Tumor Necrosis Factor", "type": "Protein"},
        {"entity_id": "ACE2", "name": "ACE2 Receptor", "type": "Protein"},
        {"entity_id": "Cytokine_Storm", "name": "Cytokine Storm", "type": "Phenotype"}
    ]
    
    node_ids = []
    for entity in entities:
        nid = kg_lib.add_node(entity)
        node_ids.append(nid)
        print(f"🔗 Added node: {entity['name']}")
    
    print(f"\n📊 Created {len(hypothesis_ids)} hypotheses and {len(node_ids)} nodes")

if __name__ == "__main__":
    # Run basic example
    main()
    
    # Optionally run advanced example
    print("\n" + "="*60)
    response = input("Run advanced example? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        advanced_example()
    
    print("\n🎉 BioDisco example complete! Check the documentation for more features.")
