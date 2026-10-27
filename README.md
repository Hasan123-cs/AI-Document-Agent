# AI Document Assistant

An AI-powered document analysis application built with **Python, Streamlit, and Google Gemini**. The application enables users to upload PDF and Excel documents, ask questions in natural language, and receive AI-generated answers supported by document references and structured data analysis.

## Features

- **PDF and Excel Processing:** Supports PDF, XLSX, and XLS documents from a local folder or through a Streamlit multi-file uploader.
- **Drag-and-Drop File Upload:** Users can upload multiple documents directly through the dashboard without manually adding files to the project.
- **Session-Isolated Uploads:** Each visitor's uploaded documents are processed in memory and remain accessible only within their own Streamlit session.
- **Document Management:** View uploaded files, remove individual documents, or clear uploaded documents.
- **Natural-Language Question Answering:** Ask questions about document content using Google Gemini.
- **Gemini Function Calling:** Automatically selects tools for document searching and structured Excel analysis.
- **Excel Data Analysis:** Supports `filter`, `sum`, `average`, `count`, `min`, `max`, and `list_columns`, including supported numeric threshold comparisons.
- **Deterministic Calculations:** Uses Python-based calculations for structured Excel data instead of relying solely on AI-generated arithmetic.
- **Source References:** Displays PDF page references and Excel sheet/row references where available.
- **Modern Streamlit Dashboard:** Responsive interface with document statistics, interactive chat, example questions, and structured spreadsheet results.
- **Local Search Fallback:** Returns relevant document search results when Gemini is temporarily unavailable.
- **Document Caching:** Reduces unnecessary document reprocessing.
- **Error Handling:** Handles invalid documents, empty collections, upload limits, and common API errors.
- **Automated Testing:** Uses pytest to validate document processing, tool execution, and agent behavior.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and document processing |
| Streamlit | Interactive web dashboard and file uploads |
| Google Gemini API | Natural-language processing and function calling |
| PyMuPDF | PDF text extraction |
| pandas | Structured Excel data processing |
| openpyxl | Excel workbook processing |
| Decimal | Precise numeric calculations |
| python-dotenv | Local environment configuration |
| pytest | Automated testing |
| Git & GitHub | Version control and repository hosting |
| Streamlit Community Cloud | Web deployment |
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

- **`app/main.py`** — Streamlit dashboard, file uploads, session state, document management, chat, and result presentation.
- **`app/agent.py`** — Google Gemini integration, function calling, tool orchestration, and fallback handling.
- **`app/tools.py`** — Document searching and structured Excel operations.
- **`app/document_loader.py`** — PDF and Excel extraction, chunk creation, and source metadata.
- **`app/models.py`** — Shared data models, including `DocumentChunk` and `AgentAnswer`.
- **`tests/`** — Automated tests for document processing and agent functionality.

## Installation and Setup

### Prerequisites

- Python 3.10 or newer, subject to installed dependency compatibility
- Google Gemini API key
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Hasan123-cs/AI-Document-Agent.git
cd AI-Document-Agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never commit your actual API key to GitHub.

### 5. Add Documents

You can use either method:

**Option A — Upload documents through Streamlit**

Upload PDF, XLSX, or XLS files directly from the dashboard.

- Multiple files supported.
- Maximum 10 MB per file.
- Maximum 30 MB total uploaded data per session.
- Uploaded files are processed in memory.
- Uploaded files are not written to the local `documents/` folder.

**Option B — Use the local documents folder**

Place your files inside `documents/`:

```text
documents/
├── contract.pdf
├── project_scope.pdf
└── boq.xlsx
```

Both local and uploaded documents can be used by the document analysis agent.

### 6. Run the Application

```bash
python -m streamlit run app/main.py
```

Open the local URL displayed in your terminal, typically:

`http://localhost:8501`

## How It Works

1. **Document Input:** The application loads PDF and Excel files from the local folder or through the Streamlit uploader.
2. **Document Validation:** Uploaded files are checked for supported formats and size limits.
3. **Document Extraction:** PDF text and Excel spreadsheet content are extracted.
4. **Chunk Creation:** Extracted information is converted into searchable document chunks with source metadata.
5. **Session Management:** Uploaded document chunks are associated with the current Streamlit session.
6. **User Question:** The user asks a question through the chat interface.
7. **Gemini Function Calling:** Gemini selects an appropriate tool based on the question.
8. **Tool Execution:** Python executes document searches or Excel analysis operations.
9. **Answer Generation:** Gemini uses the tool results to generate a response.
10. **Source References:** The application displays available document references.
11. **Fallback Handling:** When appropriate, verified Excel calculations or local document search results can be returned if Gemini is unavailable.

## Excel Analysis Operations

The application supports the following structured Excel operations:

| Operation | Description | Example Question |
|---|---|---|
| `list_columns` | Display available spreadsheet columns | What columns are available? |
| `filter` | Find rows matching supported conditions | Show items costing more than 1,000 |
| `sum` | Calculate the total of numeric values | What is the total cost? |
| `average` | Calculate the arithmetic mean | What is the average cost? |
| `count` | Count matching records | How many items are listed? |
| `min` | Find the minimum numeric value | What is the lowest cost? |
| `max` | Find the maximum numeric value | What is the highest cost? |

Excel calculations use Python-based structured data processing to improve numerical reliability.

## File Upload and Session Isolation

The application supports document uploads through Streamlit's native file uploader.

### Upload Features

- Upload multiple PDF and Excel documents.
- Display uploaded document information.
- Remove individual uploaded documents.
- Clear uploaded documents.
- Update the active document collection.
- Ask questions about uploaded documents.
- Display available source references.

### Session Isolation

Each Streamlit visitor has an independent session.

Uploaded document content is processed in memory and scoped to that session, helping prevent one visitor from accessing another visitor's uploaded documents.

Uploaded files are not permanently stored in the project folder.

**Note:** Files committed to the `documents/` folder are shared application resources and may be accessible to all visitors. Do not deploy private documents in this folder.

## Streamlit Community Cloud Deployment

The application is designed for deployment on Streamlit Community Cloud.

### Deployment Steps

1. Push the project to GitHub.
2. Visit https://share.streamlit.io.
3. Sign in with GitHub.
4. Select the repository containing the application.
5. Select the correct branch, usually `main`.
6. Set the main file path to:

```text
app/main.py
```

7. Open Advanced settings and configure the Gemini API key in Streamlit Secrets:

```toml
GEMINI_API_KEY = "your_actual_api_key"
```

8. Deploy the application.
9. Test document uploads, Gemini responses, and Excel calculations.

### Public Deployment Considerations

- Keep the Gemini API key private.
- Do not commit sensitive documents.
- Uploaded documents are session-scoped.
- Gemini API usage is subject to provider limits.
- Public users should avoid uploading confidential or sensitive information.
- Large document collections may require additional resource optimization.

## Example Questions

### PDF Analysis

- Summarize the uploaded PDF document.
- What are the main project requirements?
- What is the scope of work?
- Which page mentions payment conditions?
- What deliverables are described in the document?

### Excel Analysis

- What is the total cost of all items?
- Calculate the average cost.
- Which item has the highest cost?
- Show all items costing more than SAR 1,000,000.
- How many records match the selected condition?
- List the available Excel columns.

Results depend on the documents provided.

## Testing

The application includes automated tests using pytest.

Run the full test suite:

```bash
python -m pytest -v
```

Or:

```bash
python -m pytest tests
```

The test suite covers document processing, Excel analysis, agent behavior, source references, and error handling.

## Error Handling and Limitations

The application handles common errors, including:

- Unsupported file formats
- Corrupted or unreadable documents
- Empty document collections
- Invalid Excel columns
- Ambiguous spreadsheet column names
- Invalid numeric inputs
- Upload size limits
- Gemini API failures
- Tool execution limits

### Current Limitations

- Scanned PDFs without extractable text may require OCR.
- Complex PDF tables may not be extracted perfectly.
- Accuracy depends on document structure and extraction quality.
- Very large document collections may require further optimization.
- AI-generated responses depend on Gemini API availability.
- Public hosting resources and API usage limits may affect performance.

## AI-Assisted Development

This project was developed with assistance from **Cursor**, used for implementation, debugging, testing, and code refactoring.

The project demonstrates practical use of AI-assisted development alongside Python programming, document processing, LLM integration, and automated testing.

For development time, technical decisions, and future improvements, see [DEVELOPMENT_NOTE.md](DEVELOPMENT_NOTE.md).

## Project Purpose

The AI Document Assistant demonstrates:

- Python application development
- AI integration using Google Gemini
- LLM Function Calling
- PDF and Excel processing
- Structured numerical analysis
- Session-based document management
- Streamlit dashboard development
- Error handling and fallback mechanisms
- Automated testing
- Cloud deployment

The goal is to provide an accessible document analysis application that allows users to interact with PDF and Excel content through natural-language questions.
