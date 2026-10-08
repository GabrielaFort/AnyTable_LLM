# AnyTable LLM Assistant

AnyTable LLM Assistant is a Streamlit app for exploring CSV data with natural-language questions. Upload a CSV inspect the dataset, ask questions, generate plots, and run statistical analyses.

The app routes questions to focused LLM modules for table queries, plotting, statistics, and error correction. It also provides an optional plain-language explanation of generated analysis code.

## Setup

This project uses Python 3.11 and an Ollama-compatible chat endpoint.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r docker/requirements.txt
export OLLAMA_API_KEY_3="your-api-key"
# Optional: defaults to glm-5.3:cloud
export ANYTABLE_LLM_MODEL="gpt-oss:120b-cloud"
streamlit run app.py
```

An Ollama cloud api key must be supplied to access LLM models (current default: glm-5.3:cloud)

Open the local URL printed by Streamlit, choose an example dataset or upload a CSV, then ask a question about the table.

The endpoint lives in `src/utils.py`. Set `ANYTABLE_LLM_MODEL` to use a different
Ollama-compatible model; it defaults to `glm-5.3:cloud`.

## Docker

```bash
docker build -f docker/Dockerfile -t anytable-llm .
docker run --rm -p 8501:8501 \
  -e OLLAMA_API_KEY_3="your-api-key" \
  -e ANYTABLE_LLM_MODEL="gpt-oss:120b-cloud" \
  anytable-llm
```

Then open `http://localhost:8501`.

## Project layout

- `app.py` — Streamlit interface and CSV upload flow.
- `src/manager.py` — routes questions to the relevant analysis module.
- `src/analytical_modules/` — table QA, plotting, statistics, explanation, and error-correction modules.
- `src/llm_client.py` — Ollama-compatible API client.
- `docker/` — container configuration and Python dependencies.
- `test_data/` - example data for testing (irAE data is derived from FAERS and published in [JAMIA Open](https://doi.org/10.1093/jamiaopen/ooag094))


**Note**: This project is based on the code and workflow in [Python_irAE_LLM_Query](https://github.com/GabrielaFort/Python_irAE_LLM_Query) and published [here](https://doi.org/10.1093/jamiaopen/ooag094)
