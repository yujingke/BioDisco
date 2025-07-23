# BioDisco 🧬🤖

**AI-powered Biomedical Discovery Agent System**

BioDisco is a comprehensive framework for scientific hypothesis generation and biomedical literature mining using AI agents and knowledge graphs. It leverages multiple AI agents to automatically discover patterns, generate hypotheses, and gather supporting evidence from biomedical literature and knowledge databases.

## 🌟 Features

- **Multi-Agent AI System**: Coordinated AI agents for different aspects of scientific discovery
- **Hypothesis Generation**: Automated generation of novel biomedical hypotheses
- **Literature Mining**: Intelligent PubMed search and literature analysis
- **Knowledge Graph Integration**: Neo4j-based knowledge graph for storing and querying biomedical entities
- **Evidence Collection**: Systematic gathering and linking of supporting evidence
- **Deduplication**: Smart content deduplication across all data types
- **Extensible Architecture**: Modular design for easy extension and customization

## 🚀 Quick Start

### Installation

```bash
pip install biodisco
```

### Development Installation

```bash
git clone https://github.com/yourusername/BioDisco.git
cd BioDisco
pip install -e .
```

### Basic Usage

```python
from BioDisco import HypothesisLibrary, LiteratureLibrary, KGLibrary

# Initialize libraries
hypo_lib = HypothesisLibrary()
lit_lib = LiteratureLibrary()
kg_lib = KGLibrary()

# Add a hypothesis
hypothesis_id = hypo_lib.add("BRCA1 mutations increase susceptibility to DNA damage")

# Add literature evidence
literature_id = lit_lib.add({
    "pmid": "12345678",
    "title": "BRCA1 and DNA Repair",
    "abstract": "Study showing BRCA1's role in DNA repair mechanisms..."
})

# Add knowledge graph entities
node_id = kg_lib.add_node({
    "entity_id": "BRCA1_gene",
    "name": "BRCA1",
    "type": "Gene"
})
```

## 📋 Prerequisites

- Python 3.8+
- Neo4j database (for knowledge graph functionality)
- OpenAI API key (for AI agent functionality)

### Environment Setup

1. **Neo4j Setup**:
   ```bash
   # Install Neo4j and start the service
   # Default credentials: neo4j/password
   ```

2. **Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="your-password"
   ```

## 🏗️ Architecture

BioDisco consists of several core components:

### Core Libraries
- **BaseLibrary**: Foundation class for all data storage with UUID-based identification
- **HypothesisLibrary**: Storage and deduplication of scientific hypotheses
- **LiteratureLibrary**: PubMed literature management with PMID-based deduplication
- **KGLibrary**: Knowledge graph nodes and edges with relationship tracking
- **EvidenceLibrary**: Links between hypotheses and supporting evidence

### AI Agents
- **KeywordExtractorAgent**: Extracts relevant keywords from text
- **HypothesisQueryAgent**: Generates search queries for hypothesis validation
- **DomainSelectorAgent**: Identifies relevant scientific domains
- **ScientistAgent**: Orchestrates the discovery process

### Data Integration
- **Neo4j Integration**: Graph database for complex biomedical relationships
- **PubMed API**: Automated literature search and retrieval
- **Embedding Models**: Semantic similarity and search capabilities

## 📖 Detailed Usage

### Working with Hypotheses

```python
from BioDisco import HypothesisLibrary, EvidenceLibrary

# Create hypothesis library
hypo_lib = HypothesisLibrary()

# Add hypotheses (automatic deduplication)
h1 = hypo_lib.add("Vitamin D deficiency increases COVID-19 severity")
h2 = hypo_lib.add("Vitamin D deficiency increases COVID-19 severity")  # Same as h1
assert h1 == h2  # Returns same ID for duplicate content

# Get hypothesis
hypothesis_text = hypo_lib.get(h1)
print(hypothesis_text)
```

### Literature Management

```python
from BioDisco import LiteratureLibrary

lit_lib = LiteratureLibrary()

# Add literature with automatic PMID-based deduplication
paper = {
    "pmid": "33234567",
    "title": "COVID-19 and Vitamin D: A Systematic Review",
    "abstract": "This systematic review examines...",
    "authors": ["Smith, J.", "Doe, A."],
    "journal": "Nature Medicine",
    "year": 2021
}

lit_id = lit_lib.add(paper)
retrieved_paper = lit_lib.get(lit_id)
```

### Knowledge Graph Operations

```python
from BioDisco import KGLibrary

kg_lib = KGLibrary()

# Add nodes
gene_id = kg_lib.add_node({
    "entity_id": "VDR_gene",
    "name": "Vitamin D Receptor",
    "type": "Gene",
    "synonyms": ["VDR", "NR1I1"]
})

disease_id = kg_lib.add_node({
    "entity_id": "COVID19_disease",
    "name": "COVID-19",
    "type": "Disease",
    "synonyms": ["SARS-CoV-2 infection"]
})

# Add edges (relationships)
edge_id = kg_lib.add_edge({
    "source": "VDR_gene",
    "target": "COVID19_disease", 
    "relation": "associated_with",
    "evidence_strength": 0.8
})

# Retrieve all nodes and edges
all_nodes = kg_lib.all_nodes()
all_edges = kg_lib.all_edges()
```

### Evidence Linking

```python
from BioDisco import EvidenceLibrary

evidence_lib = EvidenceLibrary()

# Link hypothesis with supporting evidence
evidence_id = evidence_lib.add(
    hypothesis_id=h1,
    literature_ids=[lit_id],
    kg_node_ids=[gene_id, disease_id],
    kg_edge_ids=[edge_id],
    prev_hypothesis_ids=[]  # Can reference previous hypotheses
)
```

## 🔧 Configuration

### LLM Configuration

```python
from BioDisco.llm_config import gpt4o_mini_config

# Customize AI model settings
custom_config = {
    "chat_model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_output_tokens": 1500,
    "timeout": 300000
}
```

### Neo4j Configuration

```python
from BioDisco.neo4j_query import Neo4jGraph

# Initialize with custom settings
neo4j_graph = Neo4jGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password"
)
```

## 📚 Examples

Check out the `/examples` directory for comprehensive examples:

- `basic_usage.py`: Getting started with core functionality
- `hypothesis_pipeline.ipynb`: Full hypothesis generation pipeline
- `literature_analysis.ipynb`: Advanced literature mining techniques
- `knowledge_graph_demo.ipynb`: Working with biomedical knowledge graphs

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=BioDisco --cov-report=html
```

## 📖 Documentation

Full documentation is available at: [https://biodisco.readthedocs.io/](https://biodisco.readthedocs.io/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the foundation AI models
- Neo4j for graph database technology
- PubMed/NCBI for biomedical literature access
- The scientific community for open data and collaboration

## 📧 Contact

- **Team**: BioDisco Team
- **Email**: contact@biodisco.ai
- **Issues**: [GitHub Issues](https://github.com/yourusername/BioDisco/issues)

## 🔄 Version History

- **v0.1.0**: Initial release with core functionality
  - Basic library system for data management
  - AI agent framework
  - Neo4j and PubMed integration
  - Hypothesis generation and evidence collection

---

**Made with ❤️ for the scientific community**
