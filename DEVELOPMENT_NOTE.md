# AI Document Assistant — Development Note

**AI Coding Agent:** Cursor  
**Development Time:** Approximately 8 hours

### AI-Assisted Development
I used Cursor as my AI coding assistant to help implement, debug, and refactor the application. I also reviewed the code to understand its functionality, architecture, and integration between components.

### Key Design Decisions
I chose **Python and Streamlit** to create a lightweight application with a simple web interface. I integrated **Google Gemini with function calling** to process natural-language questions and select appropriate tools for document searching and Excel analysis.

I separated document loading, tool execution, AI orchestration, and the user interface into independent modules to improve code readability and maintainability. I also implemented document caching to reduce unnecessary processing, source references for traceability, and a local search fallback when the AI service is unavailable.

### Future Improvements
Given more time, I would implement semantic search using embeddings, add OCR support for scanned PDFs, improve extraction from complex tables, and expand automated testing with larger and more diverse documents.