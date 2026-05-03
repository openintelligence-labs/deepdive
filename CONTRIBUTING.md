# Contributing to DeepDive

Thanks for your interest!

## Dev setup

```bash
git clone https://github.com/openintelligence-labs/deepdive
cd deepdive
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pip install -e ../actants  # if developing against a local actants checkout
pytest tests/
```

You'll also need SearxNG and Ollama running for end-to-end tests:

```bash
docker compose up -d ollama searxng
ollama pull llama3.2
```

## Before opening a PR

- `ruff check . && ruff format --check .`
- `pytest tests/`
- New public functions have docstrings
- New behavior has test coverage
- README is updated if public behavior changed

## Principles

- Local-first, no cloud dependencies
- Privacy: no telemetry, no analytics
- All I/O is async
- Use actants for LLM calls — don't call providers directly

Join us on [Discord](https://discord.gg/openintelligence-labs).
