FROM python:3.11-slim

WORKDIR /app

# Unbuffered stdio, so the startup guard's message always reaches `docker logs`.
#
# `get_settings()` in src/config.py refuses to start on a missing BOT_TOKEN, a missing LLM API key
# or an unknown LLM_PROVIDER by logging one error line and calling `os._exit(1)`, which skips
# interpreter shutdown and drops anything still in a userspace buffer. Container stdio is a pipe,
# which CPython block-buffers for stdout. stderr (loguru's default sink) is line-buffered and loguru
# flushes after each record, so the line arrives today even without this variable; setting it keeps
# that guarantee in this repo instead of relying on the logging stack (e.g. `enqueue=True` or a
# batching sink would break it silently, leaving an empty log on exit 1).
#
# The smoke gate in .gitea/workflows/ depends on this: it runs the image with three broken
# environments, asserts on the text of the output, and checks that this variable is present in the
# image config.
ENV PYTHONUNBUFFERED=1

RUN mkdir -p data && mkdir -p src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
COPY src/*.py src/

CMD ["python", "main.py"]
