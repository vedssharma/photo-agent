# Agent evals

A fixed set of photos and requests (`CASES` in `src/photo_agent/evals.py`) run through the
real agent loop and scored by checks that measure what each request promised. See the
module docstring for the checks.

```sh
uv run python -m photo_agent.evals --replay          # offline: the reference edits
uv run python -m photo_agent.evals                   # live: Claude edits (needs a key)
uv run python -m photo_agent.evals --judge           # plus Claude's 1-10 rating per case
uv run python -m photo_agent.evals --update-baseline # save a live run as the new bar
```

CI runs the replay suite whenever the agent changes, which keeps the checks honest: every
reference edit must score 100%. When the repository has an `ANTHROPIC_API_KEY` secret, CI
also runs the live suite and fails if it scores more than 5 points below
`baseline.json` (or a case falls more than a third below its own baseline).

`baseline.json` starts at a conservative 80% with no per-case floors, since no live run has
been recorded yet. After the first live run you trust, save it with `--update-baseline`.
