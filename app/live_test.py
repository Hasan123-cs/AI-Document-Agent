from app.models import DocumentChunk
from app.agent import ask_question

# Create a sample PDF chunk
chunks = [
    DocumentChunk(
        text="The pump pressure rating is 150 PSI.",
        filename="guide.pdf",
        source_type="pdf",
        page_number=2,
    )
]

# Ask Gemini a question
answer = ask_question(
    question="What is the pump pressure rating?",
    chunks=chunks
)

print("\nLOCAL FALLBACK:" if answer.is_fallback else "\nAI ANSWER:")
print(answer.text)

print("\nSOURCES:")
for source in answer.source_labels():
    print(source)

if answer.error:
    print("\nERROR:")
    print(answer.error)