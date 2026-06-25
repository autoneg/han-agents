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
        ├── report.pdf              # per-team report (always available for qualified agents)
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

<!-- BEGIN generated standings region -->

<!-- BEGIN generated standings: 2026 -->

## ANAC 2026 Results

### Qualified agents (22)

| # | Agent | ID | Author | Team | Institute | Country |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AdaptiveBargainNegotiator | 21709 | Hajime Endo | Team Ukku | Tokyo University of Agriculture and Technology | Japan |
| 2 | AegisAgentR168 | 22280 | Mizuno | Team 374 | — | Japan |
| 3 | Agent96 | 22147 | Ismail Kerfai | Agent96 | Leibniz Universität Hannover | Germany |
| 4 | AgoraAINegotiator | 21181 | Christos Tsoufis | agoraAI | Université Paris Dauphine - PSL | France |
| 5 | CivicHAN | 22270 | Michael Ibrahim | Team 409 | Cairo University | Egypt |
| 6 | Closerv23 | 22286 | Avinash Pathak | Team 422 | Independent | India |
| 7 | CodexAgentHan | 21723 | Ryota GENSEKI | Team 507 | Tokyo University of Agriculture and Technology | Japan |
| 8 | Equinox | 21099 | Roshia | Equinox | Kyoto University | Japan |
| 9 | Group8 | 21399 | Asim Sallio | Group_8 | Özyeğin University | Türkiye |
| 10 | Gunner_Agent | 21480 | Omer Shani Steinmetz | Team 372 | College of Management Academic Studies | Israel |
| 11 | HannariHamaguriHAN | 21125 | Rinon Asanuma | Team 376 | Tokyo University of Agriculture and Technology | Japan |
| 12 | HiHan | 21736 | カズマ | チーム298 | Tokyo University of Agriculture and Technology | Japan |
| 13 | HybridPisaNegotiator | 21787 | Beste Nur Pacci | Team 508 | Özyeğin University | Türkiye |
| 14 | LastOffer | 21405 | Felix Bieber | Last Offer | Leibniz Universität Hannover | Germany |
| 15 | NegotiatorX | 21049 | Serhat Giydiren | TeamX | Özyeğin University | Türkiye |
| 16 | Nekotiator | 21656 | Toshikazu Ogura | Team 390 | Nagoya Institute of Technology | Japan |
| 17 | NeoNegotiator | 21627 | Eymen | Team 377 | Özyeğin University | Türkiye |
| 18 | NEXUSNegotiator | 21113 | Shahzeen Ahmad | Team 387 | Özyeğin University | Türkiye |
| 19 | Semruk | 21086 | Mehmet Tuğberk ÇİL | Semruk | Özyeğin University | Türkiye |
| 20 | SnowyDayAgent | 22262 | Tyrone Serapio | ST | Brown University | United States |
| 21 | Sun | 21400 | Uraz Kağan GÜNEŞ | Team 434 | Özyeğin University | Türkiye |
| 22 | T2Agent | 21146 | TogasakiTakashi | Team 397 | Tokyo University of Agriculture and Technology | Japan |

Get them after install with:

```python
get_agents(2026, qualified_only=True)
```

**Disqualified (2):** hagent, MiAgent

<!-- END generated standings: 2026 -->

<!-- END generated standings region -->
