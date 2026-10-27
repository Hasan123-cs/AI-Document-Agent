# AI Document Assistant

An AI-powered document analysis application built with **Python, Streamlit, and Google Gemini**. The application reads PDF and Excel files from a local folder and answers natural-language questions about their contents using LLM function calling and dedicated document analysis tools.

Developed as a technical assessment for the **Junior Software & IT Support** role.

## Features

- **PDF and Excel Processing:** Reads PDF, XLSX, and XLS documents from the local folder or native Streamlit multi-file upload.
- **Session-Isolated Uploads:** Uploaded documents are processed in memory, available only to the current Streamlit session, and removed when cleared or the session ends.
- **Natural-Language Question Answering:** Users can ask questions through a simple Streamlit web interface.
- **Responsive Analysis Dashboard:** A professional, mobile-aware workspace for document status, chat, source citations, and structured spreadsheet results.
- **AI Function Calling:** Google Gemini selects appropriate tools to search documents or analyze Excel data.
- **Excel Data Analysis:** Supports filtering, threshold comparisons, sums, averages, counts, and extrema on structured spreadsheet data.
- **Source References:** Displays references to the documents used to answer questions.
- **Error Handling:** Handles empty folders, unreadable files, and common API errors.
- **Local Search Fallback:** Returns relevant document search results when Gemini is temporarily unavailable.
- **Document Caching:** Reduces repeated PDF and Excel processing.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend logic and document processing |
| Streamlit | Web user interface |
| Google Gemini API | Natural-language processing and function calling |
| PDF and Excel parsing libraries | Reading and extracting document content |
| Pytest | Automated testing |
| Cursor | AI-assisted development |

## Project Structure

```text
AI-Document-Agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   ├── tools.py
│   ├── document_loader.py
│   └── models.py
├── documents/
├── tests/
├── requirements.txt
├── README.md
└── DEVELOPMENT_NOTE.md
```

### Main Components

- **`app/main.py`** — Streamlit interface, document status, question submission, answer display, and caching.
- **`app/agent.py`** — Gemini integration, function calling, tool orchestration, and fallback handling.
- **`app/tools.py`** — Local document searching and structured Excel analysis.
- **`app/document_loader.py`** — PDF and Excel extraction, chunk creation, and source metadata.
- **`app/models.py`** — Shared data models, including `DocumentChunk` and `AgentAnswer`.
- **`tests/`** — Automated tests for application functionality.

## Installation and Setup

### Prerequisites

- Python 3.10 or newer (verify compatibility with the installed dependencies)
- A Google Gemini API key
- Git (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/Hasan123-cs/AI-Document-Agent
cd AI-Document-Agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
```

`GEMINI_MODEL` is optional; the default is `gemini-3.1-flash-lite`. The agent
uses the configured model as-is and does not silently switch to another model.
Never commit your actual API key to GitHub.

### 5. Add Documents

Place your PDF and Excel files inside the `documents/` directory.
You can also upload multiple PDF, XLSX, or XLS files from the dashboard. Uploads are
limited to 10 MB per file and 30 MB total per session, processed in memory, and not
written to the documents folder.

Example:

```text
documents/
├── contract.pdf
├── project_scope.pdf
└── boq.xlsx
```

### 6. Run the Application

From the project root, execute:

```bash
python -m streamlit run app/main.py
```

Streamlit will display a local URL where you can access the application.

## How It Works

1. The application reads PDF and Excel files from the documents folder.
2. The user may additionally upload PDF and Excel documents; each upload is validated and converted to source-aware chunks in the current Streamlit session.
3. Extracted local and uploaded content is combined into searchable document chunks with source metadata.
4. The user submits a natural-language question through the Streamlit interface.
5. The Gemini agent uses function calling to select an appropriate document analysis tool.
6. The selected tool searches document content or performs structured Excel analysis.
7. The agent returns an answer with available source references.
8. If Gemini is unavailable, the application can return local search results as a fallback.

### Streamlit Community Cloud

Deploy the repository using `app/main.py` as the app entry point, and configure
`GEMINI_API_KEY` and (optionally) `GEMINI_MODEL` in the app's Streamlit secrets.
For example:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
GEMINI_MODEL = "gemini-3.1-flash-lite"
```

The Streamlit entry point reads these settings and makes them available to the
agent without coupling the agent module to Streamlit. The model name is used
exactly as configured; authentication, model-access, quota, network, and
availability failures are diagnosed separately. Temporary Gemini service
failures retain the local-search fallback. The uploader uses session state rather than
shared or permanent upload storage, so each visitor's uploaded chunks are scoped
to that visitor's Streamlit session. Local files in `documents/` remain available
to every visitor; do not put private documents in that folder on a public deploy.

## Example Questions

The following questions illustrate the intended capabilities. Results depend on the documents provided.

**PDF Questions**
- Summarize the scope of work described in the contract.
- What are the main project requirements?
- What deliverables are mentioned in the documents?

**Excel Questions**
- List all cost items above SAR 1 million.
- Which items have the highest costs?
- What is the total cost of the listed items?
- What is the average cost?

## Testing

Run the automated tests from the project root:

```bash
python -m pytest tests
```

The test suite is intended to validate document processing, tool behavior, and error handling.

## Error Handling and Limitations

The application includes handling for common failures, including unreadable documents, empty folders, and temporary AI service issues.

Current limitations include:

- Scanned PDFs without extractable text may require OCR.
- Complex PDF tables may not be extracted perfectly.
- Accuracy depends on the quality and structure of the provided documents.
- Very large document collections may require additional processing optimizations.
- AI-generated answers depend on the availability of the configured Gemini API.


## AI-Assisted Development

This project was developed with assistance from **Cursor**, which was used for implementation, debugging, and code refactoring.

For development time, key design decisions, and potential improvements, see [`DEVELOPMENT_NOTE.md`](DEVELOPMENT_NOTE.md).

## Project Purpose

This application demonstrates AI-assisted software development, LLM tool calling, document processing, structured data analysis, and a practical user interface for interacting with project documents.





