FROM python:3.11-slim

WORKDIR /app

# Unbuffered stdio, and it is not a convenience — it is what makes the startup guard audible.
#
# `get_settings()` in src/config.py refuses to start on a missing BOT_TOKEN, a missing LLM API key
# or an unknown LLM_PROVIDER by writing one `logger.error(...)` line and then calling
# `os._exit(1)`. That call is the problem: `os._exit` terminates the process immediately through
# the OS, skipping interpreter shutdown entirely — no atexit handlers, no `sys.stderr.flush()` on
# the way out. Anything still sitting in a userspace buffer at that moment is gone with the
# process.
#
# And a container's stdout/stderr are a PIPE, not a terminal, which is exactly the case CPython
# block-buffers by default. So whether that one line survives comes down to whether it happened to
# be flushed the instant it was written — a property of the logging library and of the
# interpreter's defaults, not of this repo. Today loguru's stream sink does flush after every
# record and CPython keeps stderr line-buffered, so the message does get out; but that is two
# third parties' current behaviour, and either could change under us: a `logger.add(...,
# enqueue=True)` moves writes to another process, a different sink may batch, and none of that
# would show up as anything but a container that exited 1 with a completely empty log. An operator
# would then be looking at a bot that is down, with no clue that a single variable was dropped from
# the stack.
#
# PYTHONUNBUFFERED=1 removes the question. It is the one place the answer can be pinned by this
# repo rather than inherited, so the guard's message reaches `docker logs` no matter what the
# logging stack does above it.
#
# The smoke gate in .gitea/workflows/ depends on precisely this: it runs the image three times with
# three different broken environments and asserts on the TEXT of what comes out, not merely on the
# exit code — because a non-zero exit alone is also what a typo in an import produces. That gate
# additionally checks that this very variable is present in the image's config, so it cannot be
# dropped without the check that relies on it going red first.
ENV PYTHONUNBUFFERED=1

RUN mkdir -p data && mkdir -p src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
COPY src/*.py src/

CMD ["python", "main.py"]
