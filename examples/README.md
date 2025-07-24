# BioDisco Examples

This directory contains examples demonstrating how to use BioDisco for biomedical discovery and research.

## 📁 Available Examples

### 1. Basic Usage (`basic_usage.py`)
A comprehensive Python script demonstrating:
- Creating and managing hypotheses
- Literature management with deduplication
- Knowledge graph operations (nodes and edges)
- Evidence linking between hypotheses and supporting data
- Logging and data export

**Run with:**
```bash
python examples/basic_usage.py
```

### 2. Hypothesis Pipeline (`hypothesis_pipeline.ipynb`)
A Jupyter notebook showcasing:
- Complete hypothesis generation workflow
- Background information processing
- Keyword extraction and management
- Knowledge graph construction
- Literature integration
- Evidence linking and analysis
- Data visualization preparation
- Results export

**Run with:**
```bash
jupyter notebook examples/hypothesis_pipeline.ipynb
```

## 🚀 Getting Started

1. **Install BioDisco**:
   ```bash
   pip install biodisco
   ```

2. **Install Jupyter (for notebook examples)**:
   ```bash
   pip install jupyter
   ```

3. **Set up environment variables** (optional):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="your-password"
   ```

## 📋 Example Topics Covered

### Core Functionality
- **Data Management**: Using BioDisco libraries for different data types
- **Hypothesis Generation**: Creating and storing scientific hypotheses
- **Literature Mining**: Managing PubMed papers and research articles
- **Knowledge Graphs**: Building biomedical entity relationships
- **Evidence Linking**: Connecting hypotheses to supporting evidence

### Advanced Features
- **Deduplication**: Automatic content deduplication across all data types
- **UUID Management**: Unique identification for all stored entities
- **Cross-referencing**: Linking related scientific concepts
- **Data Export**: Saving results for further analysis

## 🔬 Research Domains

The examples cover various biomedical research areas:

- **Neuroscience**: Alzheimer's disease, neuroinflammation
- **Infectious Disease**: COVID-19, viral infections
- **Genetics**: Gene-disease associations, genetic risk factors
- **Immunology**: Immune system function, inflammatory responses
- **Metabolism**: Insulin resistance, glucose metabolism

## 📊 Expected Outputs

Running the examples will generate:

1. **Console Output**: Progress updates and summaries
2. **JSON Files**: Exported data in structured format
3. **Log Files**: Agent activity logs (run_log.jsonl)
4. **Visualizations**: Data prepared for graph visualization

### Sample Output Files
- `biodisco_hypotheses.json` - Generated hypotheses
- `biodisco_knowledge_graph.json` - KG nodes and edges
- `biodisco_evidence.json` - Evidence links
- `run_log.jsonl` - Activity logs

## 🛠️ Customization

You can modify the examples to work with your own data:

### Using Your Own Hypotheses
```python
from BioDisco import HypothesisLibrary

hypo_lib = HypothesisLibrary()
your_hypothesis = "Your research hypothesis here"
hypothesis_id = hypo_lib.add(your_hypothesis)
```

### Adding Your Literature
```python
from BioDisco import LiteratureLibrary

lit_lib = LiteratureLibrary()
your_paper = {
    "pmid": "your-pmid",
    "title": "Your paper title",
    "abstract": "Your abstract text",
    "authors": ["Author1", "Author2"],
    "journal": "Journal Name",
    "year": 2024
}
literature_id = lit_lib.add(your_paper)
```

### Building Your Knowledge Graph
```python
from BioDisco import KGLibrary

kg_lib = KGLibrary()

# Add your entities
node_id = kg_lib.add_node({
    "entity_id": "YOUR_ENTITY",
    "name": "Entity Name", 
    "type": "Gene|Protein|Disease|Compound",
    "description": "Description of your entity"
})

# Add relationships
edge_id = kg_lib.add_edge({
    "source": "ENTITY_A",
    "target": "ENTITY_B", 
    "relation": "associated_with|causes|treats",
    "evidence_strength": 0.8
})
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Make sure BioDisco is properly installed: `pip install biodisco`
   - Check Python version compatibility (3.8+)

2. **API Key Issues**:
   - Set OpenAI API key if using AI agent features
   - Check API key permissions and quotas

3. **Neo4j Connection**:
   - Ensure Neo4j is running if using graph database features
   - Check connection credentials

4. **Jupyter Notebook Issues**:
   - Install Jupyter: `pip install jupyter`
   - Make sure BioDisco is accessible in the notebook kernel

### Getting Help

- Check the main [BioDisco README](../README.md)
- Review the [Contributing Guidelines](../CONTRIBUTING.md)
- Open an issue on GitHub
- Contact: contact@biodisco.ai

## 🎯 Next Steps

After running these examples, consider:

1. **Explore AI Agents**: Use BioDisco's AI agents for automated discovery
2. **Integrate PubMed**: Automatically search and retrieve literature
3. **Connect Neo4j**: Use a graph database for complex queries
4. **Build Workflows**: Create custom discovery pipelines
5. **Visualize Results**: Create interactive knowledge graph visualizations

## 📚 Additional Resources

- [BioDisco Documentation](https://biodisco.readthedocs.io/)
- [API Reference](https://biodisco.readthedocs.io/en/latest/api/)
- [GitHub Repository](https://github.com/yourusername/BioDisco)
- [Research Papers](https://github.com/yourusername/BioDisco/wiki/Papers)

---

**Happy Discovering! 🧬🤖**
