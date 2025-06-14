# SATIM GRC - Advanced Governance, Risk, and Compliance Analysis System

![GRC Analysis System](https://img.shields.io/badge/GRC-Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![TypeScript](https://img.shields.io/badge/TypeScript-4.5+-blue)
![LLM](https://img.shields.io/badge/LLM-Gemini-purple)

SATIM GRC is an advanced Governance, Risk, and Compliance analysis system that leverages Large Language Models (LLMs) and vector search to analyze policies, identify compliance gaps, and provide intelligent recommendations.

## System Architecture

The SATIM GRC system employs a multi-agent architecture with a sophisticated RAG (Retrieval-Augmented Generation) engine for contextual analysis of compliance documents.


### Core Components

1. **Multi-Agent System**
   - Collaborative agents working together to perform comprehensive analysis
   - Performance tracking and monitoring built into each agent
   - Sophisticated communication protocol for agent coordination

2. **Vector Search Engine**
   - FAISS-powered semantic search
   - International law and regulatory context enhancement
   - Hybrid search combining TF-IDF, BM25, and vector embedding approaches

3. **LLM Integration**
   - Google Gemini integration for advanced analysis
   - Context-aware prompting with domain expertise
   - Policy gap identification and recommendation generation

4. **Frontend Dashboard**
   - Modern React-based UI with Material UI
   - Interactive visualization of compliance metrics
   - Executive dashboards and detailed reports

## System Components

### 1. Agents System

The agents system is the core of the SATIM GRC platform, with specialized components for different aspects of compliance analysis:

#### Base Agent Framework
- `EnhancedBaseAgent`: Abstract base class with performance tracking, memory, and collaboration capabilities
- `EnhancedAgentCoordinator`: Orchestrates multi-agent workflows and manages agent collaboration
- `AgentCommunicationProtocol`: Manages communication between agents with request/response tracking

#### Specialized Agents
- `EnhancedPolicyComparisonAgent`: Analyzes policies against reference standards and performs gap analysis
- `IntelligentPolicyFeedbackAgent`: Generates actionable recommendations for policy improvement

### 2. RAG (Retrieval-Augmented Generation) System

The RAG system enhances LLM analysis with relevant policy content:

- `InternationalLawEnhancedRAGEngine`: Main engine that combines vector search with LLM capabilities
- `ContextBuilder`: Creates rich, structured context for policy comparison
- `DocumentLoader`: Loads and manages policy documents from JSON files
- `EnhancedVectorSearchService`: Provides vector similarity search with FAISS integration

### 3. Models

Data structures for system entities:

- `Policy` and `PolicySection`: Represent policy documents and their sections
- `ComplianceScore` and `ScoreCriteria`: Model for compliance scoring and evaluation
- `PolicySectionMatch`: Represents matches between company and reference policies

### 4. Utilities

Supporting components for system operation:

- `config.py`: System configuration management
- `logging_config.py`: Enhanced logging with performance tracking
- `rich_output.py`: Rich terminal output for analysis visualization

### 5. Frontend

Modern React-based UI for visualization and interaction:

- Material UI components for consistent design
- Interactive dashboards for compliance metrics
- Policy analysis and recommendation views

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LyesHADJAR/satim_grc.git
cd satim_grc
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Set up environment variables:
```bash
# Create a .env file in the root directory
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "GEMINI_MODEL=gemini-1.5-flash" >> .env
```

### Data Preparation

The system requires policy documents in a specific format:

1. Place your company policies in `preprocessing/policies/`
2. Place reference standards (like PCI DSS) in `preprocessing/norms/international_norms/`
3. Run the preprocessing script:
```bash
python preprocessing/process_documents.py
```

### Running the System

1. Start the backend:
```bash
python test/test.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Access the system at `http://localhost:3000`

## Analysis Workflow

The SATIM GRC system follows a sophisticated analysis workflow:

1. **Domain Discovery**: Automatically identifies relevant compliance domains in your policies
2. **Content Extraction**: Extracts domain-specific content from both company and reference policies
3. **Gap Analysis**: Identifies gaps between company policies and reference standards
4. **Coverage Assessment**: Evaluates the completeness of policy coverage
5. **Compliance Scoring**: Calculates quantitative compliance scores for each domain
6. **Recommendation Generation**: Produces actionable recommendations for policy improvement

## Configuration

System behavior can be customized through the `utils/config.py` file:

- LLM settings (model, temperature, etc.)
- Data paths for policy documents
- System parameters (timeouts, concurrency, etc.)
- Reporting options

## Contributing

Contributions to SATIM GRC are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for providing the Gemini API
- Facebook Research for FAISS vector search library
- Material-UI for the frontend components
