# han-agents

Finalist agents from the Human Agent Negotiation (HAN) league of the
[ANAC competition series](https://anac.cs.brown.edu). Mirrors the
layout of [`autoneg/anl-agents`](https://github.com/autoneg/anl-agents)
and [`autoneg/scml-agents`](https://github.com/autoneg/scml-agents)
so the same `pip install` + `from <pkg>.<year>.<team> import …`
workflow applies.

## Install

```bash
pip install git+https://github.com/autoneg/han-agents@v0.0.1
# or for local development against a clone:
pip install -e .
```

After install, each finalist is importable by its fully-qualified
dotted path, e.g.:

```python
from han_agents.han2026.team_a import MAIN_AGENT
from han_agents.han2026.team_b.strategy import TeamBAgent
```

## Layout

```
src/han_agents/
├── __init__.py
├── agents.py                       # get_agents(year, finalists_only, …)
└── han<year>/
    ├── __init__.py
    └── team_<id>/
        ├── __init__.py             # exposes MAIN_AGENT, __all__
        ├── requirements.txt        # per-team deps (optional)
        └── <strategy>.py           # the agent code
```

`get_agents(...)` in `agents.py` is the canonical registry — call it
with `year=2026, finalists_only=True` to enumerate the per-year
finalists.

## How finalists are published each year

The competition produces a `scmlweb` agent table with one row per
submitted agent. To turn the top-N entries into a release of this
package, use the automation script in the scmlweb repo:

```bash
cd ~/scmlweb
python python/set_han_finalists.py \
    --year 2026 \
    --ids 20566,21041,21089,21102,21155 \
    --han-agents-root ~/code/projects/han-agents
```

That script extracts each agent's submission zip, lays out a
`team_<id>/` subdirectory with `__init__.py` and `requirements.txt`,
updates `han_agents/han{year}/__init__.py` and the `get_agents(...)`
registry, and prints a ready-to-paste `PROLIFIC_FINALISTS=…` line.
Then:

```bash
cd ~/code/projects/han-agents
git add -A && git commit -m "han2026 finalists" && git tag v0.0.X && git push --tags
```

…and `pip install git+…@v0.0.X` from any host that needs the
finalists at runtime (HANI's venv on production, primarily).

## Reuse for future years

Add a new `han<year>/` subdir and bump the version. The
`get_agents(year=2027, …)` call will pick it up automatically.
