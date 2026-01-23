# Advanced RAG Chatbot

This project implements an advanced RAG (Retrieval-Augmented Generation) chatbot with multiple features including web search, conversational memory, advanced retrieval techniques, caching, and safety guardrails.

## Features

- **Modular Design**: Code is organized into logical modules (`retrieval`, `generation`, `guardrails`, etc.) for easy maintenance and expansion.
- **Advanced Retrieval Pipeline**:
    - **Query Transformation**: Uses `MultiQueryRetriever` to generate multiple perspectives on a user's question.
    - **Hybrid Search**: Combines vector search from **Qdrant** with real-time **web search** (Tavily).
    - **Re-ranking**: Employs `CohereRerank` to improve the relevance of retrieved documents before generation.
- **Generation**: Uses Google's `gemini-1.5-flash` model for fast and high-quality responses.
- **Conversational Memory**: Remembers the last few turns of the conversation to answer follow-up questions.
- **Caching**: Implements a semantic cache with `GPTCache` to provide instant answers to repeated questions.
- **Guardrails**: Includes basic checks for prompt injection and sensitive content to ensure safer interactions.
- **UI**: A simple and interactive UI built with **Gradio**.

## Project Structure

```
.
├── gradio_app.py           # Gradio UI
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── .env.example            # Example environment variables
├── config/
│   └── settings.py
└── src/
    ├── chatbot_core.py     # Main chatbot orchestration logic
    ├── cache/
    │   └── gpt_cache_manager.py
    ├── generation/
    │   └── gemini_generator.py
    ├── guardrails/
    │   ├── content_filter.py
    │   ├── injection_detector.py
    │   └── topic_classifier.py
    ├── memory/
    │   ├── conversation_memory.py
    │   └── session_manager.py
    └── retrieval/
        ├── enhanced_retriever.py
        ├── query_transformer.py
        ├── reranker.py
        └── web_search.py
```

## Setup and Installation (WSL)

This project uses `uv` for package management. It's a fast, modern Python package installer.

1.  **Install `uv`**:
    If you don't have `uv`, install it with:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    uv venv
    ```

3.  **Activate the Environment**:
    ```bash
    source .venv/bin/activate
    ```

4.  **Install Dependencies**:
    Install all required packages from `pyproject.toml`.
    ```bash
    uv pip install -e .
    gdown https://drive.google.com/file/d/1b-4Pj59xUhh0zpDa4qiGOykyAM1RHXmv/view?usp=sharing
    ```

4.5. clone hipporag
customize the code in the file src/misc_utils.py
```
def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[^\w\s\u00C0-\u024F]", " ", text.lower()).strip()
```

4.7. setup ngrok
```
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

ngrok config add-authtoken <token>

ngrok http 8000
```

5.  **Environment Variables**:
    Create a `.env` file in the root directory (you can copy `.env.example`) and add the necessary API keys:
    ```
    # --- LLMs and Services ---
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    COHERE_API_KEY="YOUR_COHERE_API_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY" # Needed for embeddings

    # --- Qdrant Vector Store ---
    # Leave blank if running locally
    QDRANT_URL="YOUR_QDRANT_CLOUD_URL"
    QDRANT_API_KEY="YOUR_QDRANT_CLOUD_API_KEY"
    ```
    *Note: If you are running Qdrant locally via Docker, you can leave `QDRANT_URL` and `QDRANT_API_KEY` blank.*

## Running the Application

To start the Gradio UI:
```bash
uv run python gradio_app.py
```

The application will start, and you can open the provided URL in your browser to interact with the chatbot.

## API Keys Required

To use this chatbot, you will need the following API keys:

1. **Google AI API Key**: For Gemini LLM
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Tavily API Key**: For web search
   - Sign up at [Tavily](https://tavily.com)
   - Get your API key from the dashboard

3. **Cohere API Key**: For re-ranking
   - Sign up at [Cohere](https://cohere.ai)
   - Get your API key from the dashboard

4. **OpenAI API Key**: For embeddings
   - Sign up at [OpenAI](https://platform.openai.com)
   - Create an API key

## Notes

- The application will create a local cache database (`cache.db`) to speed up repeated queries.
- If you're running Qdrant locally, make sure you have Docker installed and running.
- The guardrails are basic implementations. For production use, consider more sophisticated content filtering.

## Troubleshooting

If you encounter import errors, make sure all dependencies are installed:
```bash
uv pip install -e .
```

If you have issues with the virtual environment, try recreating it:
```bash
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e .
```
