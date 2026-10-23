"""List Gemini models and probe real function-calling availability."""

from __future__ import annotations

from google import genai
from google.genai import types

from app.agent import (
    MAX_MODEL_ATTEMPTS,
    _api_key_from_env,
    _http_options,
    _model_candidates,
)


def main() -> None:
    """Print model names and sanitized results without exposing API data."""
    api_key = _api_key_from_env()
    if not api_key:
        print("GEMINI_API_KEY is not configured.")
        return

    client = genai.Client(api_key=api_key, http_options=_http_options())
    try:
        try:
            print("Models listed for this API key:")
            for model in client.models.list():
                actions = list(getattr(model, "supported_actions", None) or [])
                if "generateContent" in actions:
                    print(f"- {model.name}")
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            print(f"Model listing failed: {type(exc).__name__}, status={status or 'unknown'}")

        declaration = types.FunctionDeclaration(
            name="probe_document_search",
            description="A no-op check that the model can request a local function.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"query": types.Schema(type=types.Type.STRING)},
                required=["query"],
            ),
        )
        for model_name in _model_candidates()[:MAX_MODEL_ATTEMPTS]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents="Call probe_document_search with query test.",
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(function_declarations=[declaration])],
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.ANY
                            )
                        ),
                        max_output_tokens=64,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=types.ThinkingLevel.LOW
                        ),
                    ),
                )
                calls = list(getattr(response, "function_calls", None) or [])
                result = (
                    "function-calling succeeded"
                    if calls and calls[0].name == declaration.name
                    else "request succeeded without a function call"
                )
                print(f"{model_name}: {result}")
            except Exception as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                print(
                    f"{model_name}: {type(exc).__name__}, "
                    f"status={status or 'unknown'}"
                )
    finally:
        client.close()


if __name__ == "__main__":
    main()
