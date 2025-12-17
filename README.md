## AI Insurance Claim Validator
This project is a high-performance, AI-powered system designed to validate insurance claims against PDF policy documents using Retrieval-Augmented Generation (RAG). It features a local LLM integration (Ollama), parallel processing for speed, and a dedicated performance monitoring suite.

### 🚀 Key Features
 - Dynamic RAG Engine: Works with any insurance policy (Health, Auto, Home, Life) by building a searchable knowledge base from uploaded PDFs.

 - Fast Claims Validation: Uses parallel processing to perform multiple policy checks (Coverage, Age, Limits, Exclusions) simultaneously.

 - Performance Tracking: Captures granular metrics including LLM call counts, retrieval times, and parallel execution durations.

 - Automated Dashboard: Visualizes system performance through interactive waterfall charts, time distribution pies, and efficiency metrics.

 - Regex-First Extraction: Optimized entity extraction for ages, amounts, and durations using fast regular expressions before involving the LLM.

### 🛠️ System Architecture

 - Backend (api_server.py): A FastAPI server that handles file uploads and coordinates the validation logic.

 - RAG Engine (rag_engine_free.py): The core logic using LangChain, ChromaDB, and HuggingFace embeddings to process documents and query the LLM.

 - Frontend (app.py): A Streamlit interface for users to upload policies, enter claim details, and view detailed results.

 - Performance Dashboard (performance_dashboard.py): A dedicated analytics tool to monitor the health and speed of the pipeline.

### 📋 Prerequisites
 - Python: 3.9+

 - Ollama: Installed and running locally.

 - LLM Model: mistral (recommended) or any compatible model pulled via Ollama.

```Bash

ollama pull mistral

```
### ⚙️ Setup & Installation
 - Install Dependencies:

```Bash

pip install fastapi streamlit langchain chromadb sentence-transformers uvicorn pypdf plotly torch

```
 - Start the Backend:

```Bash

uvicorn api_server:app --reload

```
 - Launch the Main Application:

```Bash

streamlit run app.py

```
### Launch the Performance Dashboard:

```Bash

streamlit run performance_dashboard.py

```
### 📖 Usage Flow
 - Upload Policy: Upload a PDF policy in the sidebar of the main app. The system will ingest it, create vector embeddings, and generate a summary.

 - Enter Claim: Provide claim details (manually or via pre-filled examples like "Health - Surgery" or "Auto - Accident").

 - Validate: Click "Validate Claim." The system retrieves relevant policy clauses and performs parallel validation checks.

 - Review Performance: View the "Raw Data" or "Performance" tab to see the execution metrics, or export the JSON to the Performance Dashboard for deep analysis.

### 📊 Monitoring Data Structure

The system tracks the following metrics for every run:

 - total_time: End-to-end processing duration.

 - entity_extraction_time: Time spent parsing claim details.

 - retrieval_time: Time taken to find relevant policy sections.

 - parallel_checks_time: Duration of the concurrent LLM validation tasks.

 - llm_calls: Total number of requests sent to Ollama.
