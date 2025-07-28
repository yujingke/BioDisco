# BioDisco 🧬🤖

**AI-powered Biomedical Discovery Agent System**

BioDisco is a comprehensive framework for scientific hypothesis generation and biomedical literature mining using AI agents and knowledge graphs. It leverages multiple AI agents to automatically discover patterns, generate hypotheses, and gather supporting evidence from biomedical literature and knowledge databases.

## 🌟 Features

- **Multi-Agent AI System**: Coordinated AI agents for different aspects of scientific discovery
- **Hypothesis Generation**: Automated generation of novel biomedical hypotheses
- **Literature Mining**: Intelligent PubMed search and literature analysis
- **Knowledge Graph Integration**: Neo4j-based knowledge graph for storing and querying biomedical entities
- **Evidence Collection**: Systematic gathering and linking of supporting evidence
- **Simple Python Interface**: Easy-to-use API for scientific discovery

## 🚀 Quick Start

### Installation

```bash
pip install biodisco
```

### Basic Usage

BioDisco provides a simple interface for biomedical discovery:

```python
import BioDisco

# Simple disease-based discovery
results = BioDisco.generate("Alzheimer's disease")

# Discovery with specific genes
results = BioDisco.generate("cancer", genes=["BRCA1", "BRCA2"])

# Structured input
results = BioDisco.generate({
    "disease": "diabetes",
    "genes": ["INS", "INSR"],
    "background": "Type 2 diabetes and insulin resistance"
})

# Custom parameters
results = BioDisco.generate(
    "Parkinson's disease",
    n_iterations=5,
    max_results=20,
    start_year=2020
)
```

### Advanced Usage

For more control over the discovery process:

```python
from BioDisco import (
    run_biodisco_full,
    run_full_pipeline,
    DomainSelectorAgent,
    DiseaseExplorerAgent
)

# Full pipeline with specific parameters
results = run_biodisco_full(
    disease="multiple sclerosis",
    core_genes=["HLA-DRB1", "IL7R"],
    start_year=2019,
    min_results=3,
    max_results=10,
    n_iterations=3
)

# Evidence-focused pipeline
evidence_results = run_full_pipeline(
    background="Research on autoimmune diseases",
    n_iterations=2,
    max_lit_per_hypo=8
)
    "abstract": "Study showing BRCA1's role in DNA repair mechanisms..."
}```

## 📋 Prerequisites

- Python 3.8+
- Neo4j database (optional, for knowledge graph functionality)
- OpenAI API key (for AI agent functionality)

### Environment Setup

1. **Create a `.env` file** in your project directory:
   ```bash
   # OpenAI API Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Neo4j Database Configuration (optional)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   
   # PubMed Configuration (optional)
   PUBMED_EMAIL=your_email@example.com
   ```

2. **Install dependencies**:
   ```bash
   pip install python-dotenv
   ```

### Development Installation

```bash
git clone https://github.com/yourusername/BioDisco.git
cd BioDisco
pip install -e .
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

```

## 📚 API Reference

### Main Interface

#### `BioDisco.generate(input_data, **kwargs)`

The primary interface for biomedical discovery.

**Parameters:**
- `input_data` (str or dict): Disease name or structured input
- `genes` (list, optional): List of gene names 
- `background` (str, optional): Background research context
- `start_year` (int, optional): Earliest publication year (default: 2019)
- `min_results` (int, optional): Minimum search results (default: 3)
- `max_results` (int, optional): Maximum search results (default: 10)
- `n_iterations` (int, optional): Number of discovery iterations (default: 3)
- `max_articles_per_round` (int, optional): Articles per iteration (default: 10)
- `node_limit` (int, optional): Knowledge graph node limit (default: 50)
- `direct_edge_limit` (int, optional): Knowledge graph edge limit (default: 30)

**Returns:**
- `dict`: Discovery results with hypotheses, evidence, and analysis

**Examples:**
```python
# Simple usage
results = BioDisco.generate("diabetes")

# With genes
results = BioDisco.generate("cancer", genes=["BRCA1", "BRCA2"])

# Structured input
results = BioDisco.generate({
    "disease": "Alzheimer's", 
    "genes": ["APP", "PSEN1"],
    "background": "Amyloid cascade hypothesis"
})

# Custom parameters
results = BioDisco.generate(
    "multiple sclerosis",
    n_iterations=5,
    max_results=20,
    start_year=2020
)
```

### Advanced Functions

- `run_biodisco_full(disease, core_genes, **params)`: Full pipeline with background generation
- `run_full_pipeline(background, **params)`: Evidence-focused pipeline
- `run_background_only(disease, core_genes, **params)`: Generate research background only

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
