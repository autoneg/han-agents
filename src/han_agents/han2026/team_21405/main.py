import random
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

# Add project root to sys.path so local modules like 'examples' can be imported
sys.path.insert(0, str(Path(__file__).parent))

from negmas.helpers import get_class, instantiate
from negmas.inout import Scenario, unique_name
from negmas.preferences import compare_ufuns
from negmas.preferences.generators import generate_multi_issue_ufuns
from negmas.sao import SAOMechanism
from negmas.tournaments.neg.simple import cartesian_tournament
from negmas_llm.tags import get_available_tags, get_tag_documentation
from negmas_llm.common import DEFAULT_MODELS
from hani.scenarios.trade import make_trade_scenario
from hani.scenarios.island import make_island_scenario
from hani.scenarios.grocery import make_grocery_scenario
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
import typer
from typing import Annotated

app = typer.Typer(help="ANL2026 CLI application")
DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS["ollama"]

# HANI scenario makers for generating domain-specific scenarios
HANI_SCENARIO_MAKERS = {
    "trade": make_trade_scenario,
    "island": make_island_scenario,
    "grocery": make_grocery_scenario,
}
# Default negotiator path used throughout the application
MY_NEGOTIATOR = "lastoffer.LastOfferNegotiator"

ALL_EXAMPLES = [
    "examples.llm.HAN2026LLMNegotiator",  # Full LLM-based negotiator using Ollama and Qwen3:4b-instruct
    "examples.llm_adapter.BoulwareBasedLLMNegotiator",  # LLM negotiator that uses a Boulware strategy for offer generation
    "examples.nollm.SimpleNeg",  # Simple SAO-based negotiator with no natural language
    "examples.nollm.BOANeg",  # BOA architecture negotiator with no natural language, using a frequency model for opponent modeling
    "examples.nollm_adapter.TemplateBasedAdapterNegotiator",  # Adapter that adds predefined messages to a base negotiator
]


ALL_COMPETITORS = [MY_NEGOTIATOR] + ALL_EXAMPLES


def generate_random_scenarios(
    n_scenarios: int = 1,
    n_issues: tuple[int, int] = (1, 4),
    n_values: tuple[int, int] = (5, 10),
    rational_fraction: float = 1.0,
) -> list[Scenario]:
    """Generate random negotiation scenarios."""
    scenarios = []
    for _ in range(n_scenarios):
        ufuns = generate_multi_issue_ufuns(
            n_issues=random.randint(*n_issues),
            n_values=n_values,
            ufun_names=("First", "Second"),
            rational_fractions=[rational_fraction, rational_fraction],
        )
        os = ufuns[0].outcome_space
        assert os is not None
        scenarios.append(Scenario(outcome_space=os, ufuns=ufuns))  # type: ignore

    return scenarios


def progress_callback(message: str, i: int, n: int, conf: dict | None = None) -> None:
    print(f"{i}/{n} - {message}")


def calc_scores(m: SAOMechanism) -> dict[str, dict[str, float]]:
    """Compute scores for the given agreement according the ANL 2026 rules."""

    # extract the agreement
    agreement = m.agreement

    # extract negotiator names
    negotiators = [_.__class__.__name__ for _ in m.negotiators]

    # find advantages (utility above reserved value)
    advantages = [
        float(_.ufun(agreement)) - float(_.ufun.reserved_value) if _.ufun else 0.0
        for _ in m.negotiators
    ]

    # calculate modeling accuracies
    ufuns = [_.ufun for _ in m.negotiators]
    models = [_.opponent_ufun for _ in m.negotiators]
    models.reverse()
    accuracies = [
        (1 + compare_ufuns(u, m, method="kendall")) / 2 for u, m in zip(ufuns, models)
    ]

    # normalize accuracies so that we divide one point among all negotiators with
    # negotiators with higher accuracy getting higher part of this point.
    accsum = sum(accuracies)
    if accsum > 0:
        accuracies = [_ / accsum for _ in accuracies]
    else:
        accuracies = [0] * len(negotiators)
    # we need to reverse to model deception (higher accuracy for opponent means lower deception score)
    accuracies.reverse()
    # return final scores. You can improve your score in one of three ways:
    # 1. Increase your advantage (negotiating a better deal for yourself)
    # 2. Increase your modeling accuracy (better opponent modeling)
    # 3. Decrease your opponent's accuracy (confuse their opponent modeling)
    return dict(
        zip(
            negotiators,
            (
                dict(Advavntage=adv, Deception=acc, Score=adv + acc)
                for adv, acc in zip(advantages, accuracies)
            ),
        )
    )


@app.command()
def run(
    # Scenario Loading Options
    scenario: Annotated[
        str | None,
        typer.Option(
            help="Scenario name to use. If not provided, a random scenario is selected from the scenarios folder.",
            rich_help_panel="Scenario Loading",
        ),
    ] = None,
    # Scenario Generation Options
    generate: Annotated[
        bool,
        typer.Option(
            "--generate-scenario/--no-generate-scenario",
            help="Generate a random scenario instead of loading one.",
            rich_help_panel="Scenario Generation",
        ),
    ] = False,
    scenario_type: Annotated[
        str | None,
        typer.Option(
            "--scenario-type",
            help="Generate a specific scenario type (trade, island, grocery). Implies --generate-scenario.",
            rich_help_panel="Scenario Generation",
        ),
    ] = None,
    # Negotiation Options
    agent: Annotated[
        str | None,
        typer.Option(
            help=f"Agent class to test. If not provided, uses {MY_NEGOTIATOR}.",
            rich_help_panel="Negotiation",
        ),
    ] = None,
    opponent: Annotated[
        str | None,
        typer.Option(
            help="Opponent class to negotiate against. If not provided, a random opponent is selected.",
            rich_help_panel="Negotiation",
        ),
    ] = None,
    # Output Options
    verbose: Annotated[
        bool,
        typer.Option(
            help="Show full negotiation trace with utilities.",
            rich_help_panel="Output",
        ),
    ] = False,
    plot: Annotated[
        bool,
        typer.Option(
            help="Plot the negotiation trace.",
            rich_help_panel="Output",
        ),
    ] = True,
    show_trace: Annotated[
        bool,
        typer.Option(
            help="Display full trace as a formatted table in terminal.",
            rich_help_panel="Output",
        ),
    ] = False,
    export_trace: Annotated[
        str | None,
        typer.Option(
            help="Export full trace to CSV file (provide file path).",
            rich_help_panel="Output",
        ),
    ] = None,
    trace_browser: Annotated[
        bool,
        typer.Option(
            "--trace-browser/--no-trace-browser",
            help="Open full trace in browser as an interactive table.",
            rich_help_panel="Output",
        ),
    ] = True,
    steps: Annotated[
        int,
        typer.Option(
            "--steps",
            help="Number of allowed negotiation steps. 0 for unlimited",
            rich_help_panel="Negotiation",
        ),
    ] = 0,
    seconds: Annotated[
        int,
        typer.Option(
            "--seconds",
            help="Number of allowed negotition seconds. 0 for unlimited",
            rich_help_panel="Negotiation",
        ),
    ] = 300,
):
    """Run a single negotiation against an opponent."""
    # set time-limits
    if seconds <= 0 and steps <= 0:
        print(
            "[red]You must either pass a nonzero time-limit (--time-limit) or a non-zero number of steps (--steps).[/red]"
        )
        exit(1)
    time_limit = None if seconds <= 0 else seconds
    n_steps = None if steps <= 0 else steps
    # Handle scenario type generation (implies --generate-scenario)
    if scenario_type or generate:
        if scenario_type:
            scenario_type_lower = scenario_type.lower()
            if scenario_type_lower not in HANI_SCENARIO_MAKERS:
                print(
                    f"[red]Unknown scenario type: {scenario_type}. "
                    f"Available types: {', '.join(HANI_SCENARIO_MAKERS.keys())}[/red]"
                )
                raise typer.Exit(1)
        else:
            # Randomly select a scenario type
            scenario_type_lower = random.choice(list(HANI_SCENARIO_MAKERS.keys()))
        maker = HANI_SCENARIO_MAKERS[scenario_type_lower]
        s = maker(index=random.randint(0, 9999))
        print(f"[blue]Generated {scenario_type_lower} scenario[/blue]")
    elif scenario is None:
        scenario = random.choice(
            list((Path(__file__).parent / "scenarios").iterdir())
        ).name
        print(
            f"Will use scenario [blue]{scenario}[/blue] (to select a specific scenario pass --scenario=<scenario_name>)"
        )
        s = Scenario.load(
            Path(__file__).parent / "scenarios" / scenario, ignore_discount=True
        )
        assert s is not None, f"Could not load scenario {scenario}"
    else:
        s = Scenario.load(
            Path(__file__).parent / "scenarios" / scenario, ignore_discount=True
        )
        assert s is not None, f"Could not load scenario {scenario}"

    # Set default agent if not provided
    if agent is None:
        agent = MY_NEGOTIATOR
        print(
            f"Will use agent [blue]{agent}[/blue] (to select a specific agent pass --agent=<agent_class>)"
        )

    if opponent is None:
        opponent = random.choice(ALL_EXAMPLES)
        print(
            f"Will use opponent [green]{opponent}[/green] (to select a specific opponent pass --opponent=<opponent_class>)"
        )

    opp_name = opponent.split(".")[-1]  # type: ignore
    agent_name = agent.split(".")[-1]  # type: ignore

    # Check if any LLM-based negotiators are involved and warn about duration
    llm_keywords = ["llm", "LLM", "Llm"]
    has_llm = any(any(kw in n for kw in llm_keywords) for n in [agent, opponent])
    if has_llm:
        print(
            "[yellow bold]Warning:[/yellow bold] [yellow]LLM-based negotiators detected. "
            "The negotiation may take a minute or more to complete, "
            "depending on the LLM inference speed.[/yellow]"
        )

    m = SAOMechanism(
        n_steps=n_steps, time_limit=time_limit, outcome_space=s.outcome_space
    )

    # Prepare kwargs for negotiators (if they support verbose)
    negotiator_kwargs = {}
    if verbose:
        negotiator_kwargs["verbose"] = True

    # Try to add opponent with verbose, fall back without if not supported
    try:
        m.add(
            instantiate(
                opponent,
                ufun=s.ufuns[0],
                id=opp_name,
                name=opp_name,
                **negotiator_kwargs,
            ),
        )
    except TypeError:
        # Negotiator doesn't support verbose parameter
        m.add(
            instantiate(opponent, ufun=s.ufuns[0], id=opp_name, name=opp_name),
        )

    # Try to add agent with verbose, fall back without if not supported
    try:
        m.add(
            instantiate(
                agent,
                ufun=s.ufuns[1],
                id=agent_name,
                name=agent_name,
                **negotiator_kwargs,
            )
        )
    except TypeError:
        # Negotiator doesn't support verbose parameter
        m.add(instantiate(agent, ufun=s.ufuns[1], id=agent_name, name=agent_name))

    m.run()

    # Get the trace dataframe
    trace_df = m.full_trace_with_utils_df()

    # Handle trace display options
    if verbose:
        print(trace_df)

    if show_trace:
        # Display trace as a rich table in terminal
        console = Console()
        table = Table(title="Negotiation Trace", show_lines=True)

        # Add columns
        for col in trace_df.columns:
            table.add_column(str(col), style="cyan")

        # Add rows
        for _, row in trace_df.iterrows():
            table.add_row(*[str(val) for val in row])

        console.print(table)

    if export_trace:
        # Export trace to CSV
        export_path = Path(export_trace)
        trace_df.to_csv(export_path, index=False)
        print(f"[green]Trace exported to {export_path}[/green]")

    if trace_browser:
        # Create an HTML table and open in browser
        # Exclude 'data' column from the trace display
        trace_df_display = trace_df.drop(columns=["data"], errors="ignore")

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Negotiation Trace</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th {{
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        .container {{
            max-width: 100%;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>Negotiation Trace</h1>
    <div class="container">
        {trace_df_display.to_html(index=False, classes="trace-table", border=0)}
    </div>
</body>
</html>
"""
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html_content)
            temp_path = f.name

        # Open in browser
        webbrowser.open(f"file://{temp_path}")
        print("[green]Trace opened in browser[/green]")

    if plot:
        m.plot()
    print(f"Agreement: {m.agreement}")
    print(calc_scores(m))


@app.command()
def gui(
    agents: Annotated[
        str,
        typer.Option(
            help=f"Agent class to use in GUI. Defaults to file:{MY_NEGOTIATOR}.",
            rich_help_panel="Agent",
        ),
    ] = f"file:{MY_NEGOTIATOR}",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--no-verbose",
            help="Enable verbose output for negotiators (if supported).",
            rich_help_panel="Agent",
        ),
    ] = True,
    use_dev: Annotated[
        bool,
        typer.Option(
            "--use-dev",
            help="Use 'hani --dev' instead of 'hani-guest' (recommended if you get errors).",
            rich_help_panel="Mode",
        ),
    ] = False,
    port: Annotated[
        int,
        typer.Option(
            help="Port to run the server on.",
            rich_help_panel="Server",
        ),
    ] = 5008,
    address: Annotated[
        str,
        typer.Option(
            help="Address to bind the server to.",
            rich_help_panel="Server",
        ),
    ] = "localhost",
    show: Annotated[
        bool,
        typer.Option(
            "--show/--no-show",
            help="Automatically open the app in a browser.",
            rich_help_panel="Server",
        ),
    ] = True,
    autoreload: Annotated[
        bool,
        typer.Option(
            "--autoreload/--no-autoreload",
            help="Automatically reload the server when source files change.",
            rich_help_panel="Server",
        ),
    ] = False,
):
    """Launch the HAN GUI with specified agent in guest/dev mode (no authentication)."""

    # Build the command with all options
    if use_dev:
        cmd = ["hani", "--dev", "--agents", agents]
        print(f"[blue]Launching HANI in dev mode with agent: {agents}[/blue]")
    else:
        cmd = ["hani-guest", "--agents", agents]
        print(f"[blue]Launching HANI Guest GUI with agent: {agents}[/blue]")

    # Add verbose flag
    if verbose:
        cmd.append("--verbose")

    # Add panel serve options (passed through to panel serve)
    cmd.extend(["--port", str(port)])
    cmd.extend(["--address", address])

    if show:
        cmd.append("--show")
        print("[dim]Opening in your browser...[/dim]")

    if autoreload:
        cmd.append("--autoreload")

    print(f"[dim]Server: http://{address}:{port}[/dim]\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )
        print("\n[green]HANI GUI closed successfully[/green]")
    except subprocess.CalledProcessError:
        print("\n[red]Error running HANI.[/red]")
        if not use_dev:
            print(
                "[yellow]Tip: Try using --use-dev flag if you get 'empty sequence' errors:[/yellow]"
            )
            print(f"  [green]han2026 gui --use-dev --agents {agents}[/green]")
        else:
            print(
                f"[yellow]Try running directly: [/yellow][green]hani --dev --agents {agents}[/green]"
            )
        raise typer.Exit(1)
    except FileNotFoundError:
        print(f"[red]Error: '{cmd[0]}' command not found.[/red]")
        print("\n[yellow]Installation:[/yellow]")
        print(
            "  [green]uv pip install 'hani @ git+https://github.com/autoneg/hani.git@main'[/green]"
        )
        print("\n[yellow]Alternative:[/yellow]")
        print(
            f"  Use the main hani command: [green]hani --dev --agents {agents}[/green]"
        )
        raise typer.Exit(1)


SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def load_scenarios(scenario_paths: list[str] | None) -> list[Scenario]:
    """Load scenarios from names or full paths. If None, load all from scenarios directory."""
    if scenario_paths is None:
        paths = list(SCENARIOS_DIR.iterdir())
    elif len(scenario_paths) == 1 and scenario_paths[0].lower() == "all":
        paths = list(SCENARIOS_DIR.iterdir())
    else:
        paths = []
        for s in scenario_paths:
            p = Path(s)
            if p.is_absolute() and p.exists():
                paths.append(p)
            else:
                # Treat as folder name under scenarios directory
                paths.append(SCENARIOS_DIR / s)
    scenarios = [Scenario.load(p, ignore_discount=True) for p in paths]
    return [s for s in scenarios if s is not None]


@app.command()
def tournament(
    # Tournament Options
    name: Annotated[
        str | None,
        typer.Option(
            help="Tournament name. If not provided, a unique name is generated.",
            rich_help_panel="Tournament",
        ),
    ] = None,
    competitor: Annotated[
        list[str] | None,
        typer.Option(
            help="Competitor classes to include. Can be specified multiple times. If not provided, uses default competitors.",
            rich_help_panel="Tournament",
        ),
    ] = None,
    # Scenario Loading Options
    scenario: Annotated[
        list[str] | None,
        typer.Option(
            help="Scenario names (folder names under scenarios) or full paths. Use 'all' to load all scenarios. Can be specified multiple times.",
            rich_help_panel="Scenario Loading",
        ),
    ] = None,
    # Scenario Generation Options
    generate: Annotated[
        int,
        typer.Option(
            "--generate-scenarios",
            help="Generate N random scenarios instead of loading.",
            rich_help_panel="Scenario Generation",
        ),
    ] = 0,
    scenario_type: Annotated[
        str | None,
        typer.Option(
            "--scenario-type",
            help="Generate scenarios of a specific type (trade, island, grocery). Used with --generate-scenarios.",
            rich_help_panel="Scenario Generation",
        ),
    ] = None,
    # Execution Options
    verbosity: Annotated[
        int,
        typer.Option(
            help="Verbosity level for tournament execution (0 = silent, higher = more verbose).",
            min=0,
            max=5,
            rich_help_panel="Execution",
        ),
    ] = 0,
    parallel: Annotated[
        bool,
        typer.Option(
            help="Run tournament in parallel mode (use all cores).",
            rich_help_panel="Execution",
        ),
    ] = False,
):
    """Run a complete tournament with all competitors."""
    if name is None:
        name = unique_name("anl", sep="")
    if competitor is None:
        competitor = ALL_COMPETITORS

    # Check if any LLM-based negotiators are involved and warn about duration
    llm_keywords = ["llm", "LLM", "Llm"]
    has_llm = any(any(kw in c for kw in llm_keywords) for c in competitor)
    if has_llm:
        print(
            "[yellow bold]Warning:[/yellow bold] [yellow]LLM-based negotiators detected. "
            "The tournament may take several minutes (or longer) to complete, "
            "depending on the number of scenarios and the LLM inference speed.[/yellow]"
        )

    path = Path.home() / "negmas" / "anl2026" / "tournaments" / name

    # Load scenarios from paths if provided, otherwise load all from scenarios directory
    if scenario is not None:
        scenarios = load_scenarios(scenario)
    elif generate == 0:
        scenarios = load_scenarios(None)
    else:
        scenarios = []

    # Add generated scenarios if requested
    if generate > 0:
        if scenario_type:
            scenario_type_lower = scenario_type.lower()
            if scenario_type_lower not in HANI_SCENARIO_MAKERS:
                print(
                    f"[red]Unknown scenario type: {scenario_type}. "
                    f"Available types: {', '.join(HANI_SCENARIO_MAKERS.keys())}[/red]"
                )
                raise typer.Exit(1)
            maker = HANI_SCENARIO_MAKERS[scenario_type_lower]
            generated = [maker(index=i) for i in range(generate)]
            print(f"[blue]Generated {generate} {scenario_type_lower} scenarios[/blue]")
        else:
            # Generate scenarios with randomly selected types
            generated = []
            for i in range(generate):
                scenario_type_lower = random.choice(list(HANI_SCENARIO_MAKERS.keys()))
                maker = HANI_SCENARIO_MAKERS[scenario_type_lower]
                generated.append(maker(index=i))
            print(f"[blue]Generated {generate} scenarios (mixed types)[/blue]")
        scenarios.extend(generated)

    print(f"[blue]Running tournament with {len(scenarios)} scenarios[/blue]")
    competitor_classes = [get_class(c) for c in competitor]
    results = cartesian_tournament(
        scenarios=scenarios,
        competitors=competitor_classes,  # type: ignore
        final_score=("advantage", "mean"),
        path=path,
        verbosity=verbosity,
        njobs=-1 if not parallel else 0,
        progress_callback=progress_callback,
        sort_runs=True,
        raise_exceptions=False,
    )
    print(results.final_scores)
    print(f"All Tournament Results can be found in {path}")


@app.command()
def info():
    """Show development workflows and submission instructions."""
    # Read from source directory, not installed package
    readme_path = Path(__file__).resolve().parent / "README.md"
    if not readme_path.exists():
        print("[red]README.md not found[/red]")
        raise typer.Exit(1)

    content = readme_path.read_text()

    # Sections to extract and display
    sections_to_show = [
        "## Getting Started: Rename Your Agent",
        "## Development Workflows",
        "## Submission",
    ]

    console = Console()
    for section_header in sections_to_show:
        if section_header in content:
            start = content.find(section_header)
            end = content.find("\n## ", start + 1)
            if end == -1:
                end = len(content)
            section = content[start:end].strip()
            console.print(Markdown(section))
            console.print()


@app.command()
def tags(
    tag_name: Annotated[
        str | None,
        typer.Argument(
            help="Specific tag to get documentation for. If not provided, shows all available tags.",
        ),
    ] = None,
):
    """Show available prompt tags for customizing LLM negotiator prompts.

    Tags can be used in custom prompts to dynamically insert negotiation context.
    Syntax: {{tag-name}} or {{tag-name:format(params)}}

    Examples:
      han2026 tags                    # Show all available tags
      han2026 tags utility            # Show detailed docs for 'utility' tag
      han2026 tags outcome-space      # Show detailed docs for 'outcome-space' tag
    """
    console = Console()

    if tag_name is not None:
        # Show specific tag documentation
        docs = get_tag_documentation(tag_name)
        if docs is None:
            console.print(f"[red]Unknown tag: {tag_name}[/red]")
            console.print(
                f"\n[yellow]Available tags:[/yellow] {', '.join(get_available_tags())}"
            )
            raise typer.Exit(1)
        else:
            console.print(Markdown(docs))
    else:
        # Show summary of all tags grouped by category
        console.print("\n[bold cyan]Available Tags for LLM Prompts[/bold cyan]")
        console.print("[dim]═" * 60 + "[/dim]\n")

        console.print(
            "[yellow]Usage:[/yellow] Tags use the syntax: [green]{{tag-name}}[/green] or [green]{{tag-name:format(params)}}[/green]"
        )
        console.print("[yellow]Formats:[/yellow] 'text' (default), 'json'\n")

        # Get all available tags
        all_tags = get_available_tags()

        # Group tags by category based on common patterns
        context_tags = [
            t
            for t in all_tags
            if t
            in [
                "outcome-space",
                "utility-function",
                "opponent-utility-function",
                "nmi",
                "current-state",
            ]
        ]
        reserved_tags = [t for t in all_tags if "reserved-value" in t]
        offer_tags = [t for t in all_tags if "offer" in t and "reserved" not in t]
        history_tags = [
            t
            for t in all_tags
            if t
            in ["partner-offers", "history", "trace", "extended-trace", "full-trace"]
        ]
        utility_tags = [t for t in all_tags if t == "utility"]

        # Display tags by category
        def display_tag_group(title: str, tags: list[str]):
            if tags:
                console.print(f"\n[bold]{title}:[/bold]")
                for tag in tags:
                    # Get first line of documentation as short description
                    docs = get_tag_documentation(tag)
                    short_desc = ""
                    if docs:
                        # Extract description from markdown
                        lines = docs.split("\n")
                        for line in lines[1:]:  # Skip the header line
                            line = line.strip()
                            if (
                                line
                                and not line.startswith("#")
                                and not line.startswith("**")
                            ):
                                short_desc = line
                                break
                    console.print(f"  [green]{{{{{tag}}}}}[/green]  - {short_desc}")

        display_tag_group("Context Tags (no parameters)", context_tags)
        display_tag_group("Reserved Value Tags", reserved_tags)
        display_tag_group("Offer Reference Tags", offer_tags)
        display_tag_group("History Tags (optional k parameter)", history_tags)
        display_tag_group(
            "Utility Computation Tags (requires outcome parameter)", utility_tags
        )

        console.print(
            "\n[dim]To see detailed documentation for any tag, run:[/dim] [cyan]han2026 tags <tag-name>[/cyan]"
        )
        console.print("[dim]Example:[/dim] [cyan]han2026 tags utility[/cyan]\n")


@app.command()
def setup_ollama():
    f"""Install Ollama and pull the required {DEFAULT_OLLAMA_MODEL} model.

    This command will:
    1. Check if Ollama is already installed
    2. Install Ollama if needed (platform-specific)
    3. Pull the {DEFAULT_OLLAMA_MODEL} model required for HAN 2026

    Note: On Linux, this requires curl. On macOS, this uses the official installer.
    On Windows, this will open the download page for manual installation.
    """
    console = Console()
    import shutil
    import platform

    system = platform.system().lower()

    # Check if ollama is already installed
    ollama_path = shutil.which("ollama")

    if ollama_path:
        console.print(f"[green]Ollama is already installed at: {ollama_path}[/green]")
    else:
        console.print("[yellow]Ollama not found. Installing...[/yellow]")

        if system == "linux":
            console.print("[blue]Installing Ollama on Linux...[/blue]")
            try:
                result = subprocess.run(
                    ["curl", "-fsSL", "https://ollama.com/install.sh"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Pipe to sh
                subprocess.run(
                    ["sh"],
                    input=result.stdout,
                    text=True,
                    check=True,
                )
                console.print("[green]Ollama installed successfully![/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to install Ollama: {e}[/red]")
                console.print(
                    "[yellow]Please install manually: curl -fsSL https://ollama.com/install.sh | sh[/yellow]"
                )
                raise typer.Exit(1)
            except FileNotFoundError:
                console.print("[red]curl not found. Please install curl first.[/red]")
                console.print(
                    "[yellow]On Ubuntu/Debian: sudo apt install curl[/yellow]"
                )
                raise typer.Exit(1)

        elif system == "darwin":
            console.print("[blue]Installing Ollama on macOS...[/blue]")
            # Check if brew is available
            brew_path = shutil.which("brew")
            if brew_path:
                console.print("[dim]Using Homebrew to install Ollama...[/dim]")
                try:
                    subprocess.run(["brew", "install", "ollama"], check=True)
                    console.print("[green]Ollama installed successfully![/green]")
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Failed to install via Homebrew: {e}[/red]")
                    console.print(
                        "[yellow]Please download from: https://ollama.com/download[/yellow]"
                    )
                    raise typer.Exit(1)
            else:
                console.print(
                    "[yellow]Homebrew not found. Opening download page...[/yellow]"
                )
                webbrowser.open("https://ollama.com/download")
                console.print(
                    "[yellow]Please download and install Ollama from the opened page,[/yellow]"
                )
                console.print(
                    "[yellow]then run this command again to pull the model.[/yellow]"
                )
                raise typer.Exit(0)

        elif system == "windows":
            console.print("[blue]Opening Ollama download page for Windows...[/blue]")
            webbrowser.open("https://ollama.com/download")
            console.print(
                "[yellow]Please download and install Ollama from the opened page,[/yellow]"
            )
            console.print(
                "[yellow]then run this command again to pull the model.[/yellow]"
            )
            raise typer.Exit(0)

        else:
            console.print(f"[red]Unsupported platform: {system}[/red]")
            console.print(
                "[yellow]Please install Ollama manually from: https://ollama.com/download[/yellow]"
            )
            raise typer.Exit(1)

    # Verify ollama is now available
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        console.print(
            "[red]Ollama installation completed but 'ollama' command not found.[/red]"
        )
        console.print(
            "[yellow]You may need to restart your terminal or shell and run the 'setup-ollama' again.[/yellow]"
        )
        raise typer.Exit(1)

    # Check if ollama service is running, start if needed
    console.print("\n[blue]Checking Ollama service...[/blue]")
    try:
        # Try to list models to see if service is running
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            console.print("[yellow]Ollama service not running. Starting...[/yellow]")
            # Start ollama serve in background
            if system == "windows":
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=0x00000010,  # CREATE_NEW_CONSOLE
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            console.print("[dim]Waiting for service to start...[/dim]")
            import time

            time.sleep(3)
    except subprocess.TimeoutExpired:
        console.print(
            "[yellow]Ollama service may not be running. Attempting to start...[/yellow]"
        )
        if system == "windows":
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=0x00000010,  # CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        import time

        time.sleep(3)
    except FileNotFoundError:
        console.print("[red]Ollama command not found after installation.[/red]")
        raise typer.Exit(1)

    # Pull the required model
    model_name = DEFAULT_OLLAMA_MODEL
    console.print(f"\n[blue]Pulling required model: {model_name}[/blue]")
    console.print("[dim]This may take a few minutes (2-3 GB download)...[/dim]\n")

    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
        )
        console.print(f"\n[green]Successfully pulled {model_name}![/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Failed to pull model: {e}[/red]")
        console.print(f"[yellow]Try manually: ollama pull {model_name}[/yellow]")
        raise typer.Exit(1)

    # Verify the model is available
    console.print("\n[blue]Verifying installation...[/blue]")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        if model_name.split(":")[0] in result.stdout:
            console.print(f"[green]Model {model_name} is ready to use![/green]")
            console.print(
                "\n[bold green]Setup complete![/bold green] You can now run LLM-based agents with:"
            )
            console.print("  [cyan]han2026 run[/cyan]")
            console.print("  [cyan]han2026 gui[/cyan]")
        else:
            console.print(
                "[yellow]Model pulled but not showing in list. It may still be processing.[/yellow]"
            )
    except subprocess.CalledProcessError:
        console.print(
            "[yellow]Could not verify model installation. Check the troubleshooting section in the tutorial: https://anac.cs.brown.edu/files/han/y2026/template2026.pdf[/yellow]"
        )


if __name__ == "__main__":
    app()
