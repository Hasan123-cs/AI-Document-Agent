# AI Document Assistant

Streamlit app for asking grounded questions about local PDF and Excel
documents. Gemini receives document questions and can request local tools:
`search_documents` for ranked passages and `analyze_excel` for complete
spreadsheet column discovery, numeric filtering, min/max, sums, and counts.
Python runs each requested tool and returns its results to Gemini; document
files are not uploaded to the model.

## Setup

Requirements: Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. `GEMINI_MODEL` is optional and defaults to
`gemini-3-flash-preview`. If the selected model is missing, rate-limited,
unavailable, times out, or cannot be reached, the agent tries at most one
fallback model per request. Only `generateContent` models selected for Gemini
function calling are candidates. Model-list availability does not guarantee
that a model has quota or is currently serving requests.

Place `.pdf` and `.xlsx` files in `documents/`, then start the app:

```powershell
streamlit run app/main.py
```

Use **Reload documents** after changing files. Corrupt files are logged and
skipped while valid files continue loading. An empty folder is reported in the
UI. Excel analyses operate over typed cell values from the whole workbook and
include filename, sheet, and row references. If Gemini is unavailable, the UI
shows clearly labeled local keyword matches; these are not AI answers and do
not perform spreadsheet calculations.

## Model diagnostics

The diagnostic script lists models the configured key can enumerate and makes
a small real function-calling request to the configured model and its fallback.
It prints model names and status only, never the API key or response content.
Availability and quota can change over time.

```powershell
python -m app.test
```

## Tests

```powershell
python -m pytest -q
```

Tests use local fixtures and mocked Gemini responses; they do not require live
API access.