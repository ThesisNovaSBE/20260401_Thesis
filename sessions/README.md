# sessions/

Work session logs for the readmission prediction project.

## How it works

Every time someone (human or LLM) starts a work session, they create a new file from `_TEMPLATE.md`:

```
sessions/YYYY-MM-DD_session-NN.md
```

Fill it in as you work — what you did, what decisions you made, what's broken, what comes next. Keep entries short and factual.

## For LLMs / agents

Before starting any work, read:
1. `PROJECT_TLDR.md` (project context)
2. The most recent file(s) in this directory (current state)

This tells you what was done last, what works, what's broken, and what to do next.

## Naming convention

- `YYYY-MM-DD_session-NN.md` — date + sequential number for that day
- Example: `2026-05-28_session-01.md`, `2026-05-28_session-02.md`
