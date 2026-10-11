# Agent Instructions — caldavllm_bot

Telegram bot (aiogram, long polling) that parses free-form messages and photos with an LLM and
creates events in the user's CalDAV calendar. No listening socket, no HTTP API.

## Layout
- `main.py` — entry point: builds `CalendarBot` and runs `start()`.
- `src/bot.py` — `CalendarBot`: handlers, message batching, previews.
- `src/config.py` — `get_settings()`: reads ENV (`load_dotenv()`), refuses to start
  (`os._exit(1)`) on a missing `BOT_TOKEN`, no LLM key, or an unknown `LLM_PROVIDER`.
- `src/llm.py` — picks the provider (`glm` / `groq` / `deepseek`); `src/llm_*.py` implement
  the `LLMProvider` protocol from `src/llm_base.py`.
- `src/calendar.py` — CalDAV access; `src/users.py` — per-user credentials and token limits.
- `tests/test_llm.py` — prompt-quality harness against a live, paid LLM.
- `ci/smoke.py` — hermetic smoke gate for the built image.
- `docker-compose.yml` — record of the deployed Portainer stack (placeholders only);
  README.md has a compose block for a fresh install.

## Run
```bash
pip install -r requirements.txt
BOT_TOKEN=... GLM_API_KEY=... python main.py   # or put them in .env
```
Settings: `BOT_TOKEN`, `TELEGRAM_BOT_API_SERVER`, `DEEPSEEK_API_KEY` / `GROQ_API_KEY` /
`GLM_API_KEY`, `LLM_PROVIDER` (default `glm`), `MODEL`, `TZ`, `DAILY_TOKEN_LIMIT`,
`MESSAGE_BATCH_TIMEOUT`, `MAX_BATCH_SIZE` — see `src/config.py`.

## Tests
```bash
pip install -r requirements-dev.txt
API_KEY=... pytest tests/
```
The suite calls a real LLM, costs money and asserts on answer content, so it is run by hand
when the prompt changes and never in CI.

## CI and deploy (Gitea Actions)
- `.gitea/workflows/tests.yml` — PR gate: build the image, run `python3 ci/smoke.py`. No
  secrets, no push.
- `.gitea/workflows/image-check-publish.yml` — on push to `main`: build, smoke, then push
  `:<sha>` and `:latest` to `gitea.vvzvlad.xyz`. Production follows `:latest` through
  Portainer's updater with no manual step, so the smoke gate is the last check.
- The smoke gate is the only automated check. The container-name suffixes and the timeout sum
  in `ci/smoke.py` must stay in step with both workflows.

## Invariants
- WORKDIR is `/app`, and user credentials live under the relative `data/` (mounted volume
  `/app/data`). Changing either silently moves credentials out of the volume.
- `PYTHONUNBUFFERED=1` stays in the Dockerfile: the smoke gate asserts on the guard's text.
- Most modules log through loguru (brace placeholders); `src/llm_groq.py` and
  `src/llm_glm.py` use stdlib `logging` (%-style). Do not mix the styles up.
- No secrets in code, compose or workflows; the repository is public.

## Limits for agents
- Tests: at most three times the code (`wc -l` of `tests/` against the code). Over the limit,
  still write the test you need, and in the same change go through the old tests: drop or merge
  duplicates and tests of implementation rather than behaviour.
- Comments explain the code as it is and why it is built this way. History ("used to", "after
  the review", "since the incident") goes into the commit message.
- This file stays under ~150 lines: only what an agent would get wrong without it. No decision
  history, incident logs or retelling of the code.
