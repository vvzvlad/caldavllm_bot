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
# A container's stdout and stderr are a PIPE rather than a terminal, and for STDOUT that is
# precisely the case CPython block-buffers — anything printed there before an `os._exit` is simply
# gone. stderr, which is where loguru's default sink writes, is a different story, and the honest
# version is worth writing down rather than glossed: CPython has kept stderr line-buffered
# regardless of what it is attached to since 3.9, and loguru's stream sink calls flush() after
# every record on top of that. So the guard's line does reach `docker logs` today even without
# this variable.
#
# This is therefore not a fix for a message that is currently being lost. It is about WHERE that
# guarantee lives. As things stand it rests entirely on two third parties continuing to behave as
# they do — `logger.add(..., enqueue=True)` moves writes to another process, a different or
# wrapped sink may batch, a future interpreter may revisit stderr buffering — and none of those
# changes would announce itself as anything except a container that exited 1 with a completely
# empty log. An operator would be looking at a bot that is down with no clue that a single
# variable had been dropped from the stack.
#
# PYTHONUNBUFFERED=1 moves the answer into this repo, where it can be asserted instead of assumed,
# so the guard's message reaches `docker logs` no matter what the logging stack above it does.
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
