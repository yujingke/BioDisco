# BioDisco 🧬🤖

**AI-powered Biomedical Discovery Agent System**

BioDisco is a comprehensive framework for scientific hypothesis generation and biomedical literature and knowledge graph mining using AI agents. It leverages multiple AI agents to automatically discover patterns, generate hypotheses, and gather supporting evidence from biomedical literature and knowledge databases.

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

BioDisco provides a simple interface for biomedical discovery

```python
import BioDisco

# Simple disease-based discovery
results = BioDisco.generate("Role of GPR153 in vascular injury and disease")

```

**You need to setup your Open AI API key as a environment variable `OPENAI_API_KEY`**

On your terminal 
```bash
export OPENAI_API_KEY=your_openai_api_key_here
````

or create a `.env` file in your project directory (check `.env.example`)
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
```


### Development Installation

```bash
git clone https://github.com/yujingke/BioDisco.git
cd BioDisco
pip install -e .
```

## 🔧 PubMed and Knowledge Graph Integration

By default PubMed and Knowledge Graph Integration is off. Follow the steps to setup knowledge integration.

### PubMed Setup

You can setup an an environment variable `DISABLE_PUBMED=False` in your `.env` file or using `export` command

**or**

Just pass an argument to the generate function

```python
## Turn on PubMed Integration
results = BioDisco.generate("Role of GPR153 in vascular injury and disease", disable_pubmed=False)
```

### Neo4j Setup

#### 1. Install Neo4j server 

First you need to install Neo4j server. Follow the instuctions [here](https://neo4j.com/docs/operations-manual/current/installation/) to install Neo4j for your OS

#### 2. Add Neo4j login details to as enviroment variable

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_neo4j_password
```
or set these `.env` file (check `.env.example`)

#### 3. Download and Setup PrimeKG

- Download PrimeKG 
```bash
wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620
```

- run `split_nodes_edges.py` (should be in the same location as `kg.csv`) to create `nodes.csv` and `edges.csv`

- run `build_kg_index.py` (should be in the same location as `nodes.csv`)

- add location of files as environment variable `KG_PATH`  (check `.env.example`)
```bash
export KG_PATH=/path/to/your/kg_specific_files
```

#### Import PrimeKG to Neo4j

```bash
neo4j-admin database import full --nodes nodes.csv --relationships edges.csv --overwrite-destination
```

#### Start Neo4j

```bash
neo4j start
```

#### Turn on PubMed and KG Integration

```python
results = BioDisco.generate("Role of GPR153 in vascular injury and disease", disable_pubmed=False, disable_kg=False)
```

**or**

setup environment variable `DISABLE_KG=False` (check `.env.example`)

<!-- ## 🏗️ Architecture

BioDisco consists of several core components:


### AI Agents
- **KeywordExtractorAgent**: Extracts relevant keywords from text
- **HypothesisQueryAgent**: Generates search queries for hypothesis validation
- **DomainSelectorAgent**: Identifies relevant scientific domains
- **ScientistAgent**: Orchestrates the discovery process

### Data Integration
- **Neo4j Integration**: Graph database for complex biomedical relationships
- **PubMed API**: Automated literature search and retrieval
- **Embedding Models**: Semantic similarity and search capabilities -->

## 📖 Detailed Usage

### Setting number of iterations and PubMed cutoff

```python
import BioDisco

results = BioDisco.generate("Role of GPR153 in vascular injury and disease", disable_pubmed=False, disable_kg=False, n_iterations=3, start_year=2020)
```


<!-- ## 📚 Examples

Check out the `/examples` directory for comprehensive examples:

- `basic_usage.py`: Getting started with core functionality
- `hypothesis_pipeline.ipynb`: Full hypothesis generation pipeline
- `literature_analysis.ipynb`: Advanced literature mining techniques
- `knowledge_graph_demo.ipynb`: Working with biomedical knowledge graphs -->


<!-- ## 📖 Documentation

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
- `run_background_only(disease, core_genes, **params)`: Generate research background only -->

<!-- ## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## 🙏 Acknowledgments

- OpenAI for providing the foundation AI models
- Neo4j for graph database technology
- PubMed/NCBI for biomedical literature access
- The scientific community for open data and collaboration -->

<!-- ## 📧 Contact

- **Team**: BioDisco Team
- **Email**: contact@biodisco.ai
- **Issues**: [GitHub Issues](https://github.com/yourusername/BioDisco/issues) -->

<!-- ## 🔄 Version History

- **v0.1.0**: Initial release with core functionality
  - Basic library system for data management
  - AI agent framework
  - Neo4j and PubMed integration
  - Hypothesis generation and evidence collection -->

