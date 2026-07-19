# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-07-19

### Added

- Flagged the four ANAC 2026 finalists in the registry — `AegisAgentR168`
  (22280), `CivicHAN` (22270), `Equinox` (21099), and `NegotiatorX` (21049).
  Retrieve them with `get_agents(2026, finalists_only=True)`.
- README "Finalists (4)" table listing the finalists and how to fetch them.

### Fixed

- `test_get_agents_2026_counts` now expects 4 finalists (was asserting 0),
  fixing the CI failure introduced when the finalists were flagged.

## [0.0.1] - 2026-07-18

### Added

- Initial release of the `han-agents` package: the ANAC HAN 2026 field of
  negotiation agents, packaged with a registry and `get_agents()` accessor
  (filter by `qualified_only` / `finalists_only` / `winners_only`).
- GitHub Actions CI running the agents (LLM finalists against a fake Ollama
  backend) on Python 3.11 and 3.12.
- PyPI publishing workflow using Trusted Publishing (OIDC), triggered on
  `v*` tag pushes.

[0.0.2]: https://github.com/autoneg/han-agents/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/autoneg/han-agents/releases/tag/v0.0.1
