# FPL AI Engine

Decision-support software for Fantasy Premier League squad construction and transfer planning. The engine converts official FPL player, team, fixture, and manager data into multi-gameweek expected-points projections, then solves a constrained optimisation problem for the highest-value squad trajectory available under FPL rules.

The project is designed around a practical question: **which squad and transfer sequence best balances expected points, budget, free transfers, transfer penalties, fixture difficulty, and future optionality?** It provides both an interactive Streamlit control room and a single-command command-line report.

## Capabilities

- Refreshes active player, team, fixture, and manager-state data from the official FPL API.
- Normalises the API payload into analysis-ready player and fixture datasets.
- Produces gameweek-level expected-points projections using player performance, minutes, availability, price, and fixture difficulty signals.
- Builds a season-level chip roadmap from fixture congestion, double gameweeks, blank gameweeks, and fixture difficulty.
- Solves a four-gameweek mixed-integer model-predictive-control (MPC) problem for squad state, line-up selection, transfers, free-transfer rollovers, bank balance, and hit costs.
- Ranks the resulting starting XI, captain, vice-captain, and bench for the next gameweek.
- Exposes the result through a Streamlit dashboard and an executive CLI summary.

## Architecture

```text
Official FPL API
	|
	v
Data Acquisition --> Data Validation & Normalisation --> Feature/Projection Engine
	|                         |                              |
	+-------------------------+------------------------------+
				     v
		      Season Chip Planning Layer
				     |
				     v
		    Multi-Gameweek MPC Optimisation
			    |                    |
			    v                    v
		   CLI Executive Report    Streamlit Dashboard
```

### Core modules

| Layer | Module | Responsibility |
| --- | --- | --- |
| Presentation | `app.py` | Interactive controls, pipeline execution, squad visualisation, transfer actions, chip roadmap, and analytics table. |
| Orchestration | `main.py` | Reproducible end-to-end command-line execution and reporting. |
| Acquisition | `src/fetch_data.py`, `src/fetch_fixtures.py` | Retrieve official player, team, and fixture data and persist raw exports. |
| Data processing | `src/process_data.py` | Validate, filter, normalise, and enrich player records with positions and team names. |
| Projection | `src/train_model.py` | Generate deterministic gameweek expected-points projections from player signals and fixture difficulty. |
| Season planning | `src/season_planner.py` | Identify fixture anomalies and congestion patterns and select candidate chip gameweeks. |
| Optimisation | `src/multi_week_optimizer_mpc.py` | Formulate and solve the sequential squad and transfer MILP with PuLP/CBC. |

The optimiser models the squad as a state carried from one gameweek to the next. Its objective combines discounted expected points with liquidity value and transfer-friction penalties. Constraints cover the 15-player squad structure, legal formations, three-player club limits, budget, free-transfer rollovers, and paid transfer hits.

## Technology Stack

- Python 3.10+ (type syntax and standard-library `pathlib` usage assume a modern Python runtime)
- Streamlit for the interactive application
- pandas and NumPy for tabular data processing and projection calculations
- PuLP with the CBC mixed-integer solver for constrained optimisation
- Requests for official FPL API integration
- CSV and JSON artifacts for transparent, inspectable pipeline state

## Project Layout

```text
.
├── app.py                         # Streamlit application
├── main.py                        # CLI orchestration and report
├── src/
│   ├── fetch_data.py              # Player and team acquisition
│   ├── fetch_fixtures.py          # Fixture acquisition
│   ├── process_data.py            # Data cleaning and normalisation
│   ├── train_model.py             # Multi-gameweek xP projections
│   ├── season_planner.py          # Chip roadmap generation
│   └── multi_week_optimizer_mpc.py# Four-gameweek MPC/MILP solver
└── data/                          # Raw, processed, and generated snapshots
```

## Local Setup

Create an isolated environment and install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the command-line pipeline:

```bash
python main.py
```

Launch the interactive dashboard:

```bash
streamlit run app.py
```

The pipeline requires network access to the official FPL API and writes refreshed snapshots into `data/`. The CLI uses the currently persisted manager state when available. In the dashboard, enable scratch mode to construct a new £100.0m squad, or enter an FPL team ID to load a manager's current squad before optimisation.

## Data Flow

1. `fetch_data.py` stores raw players, teams, and the detected gameweek.
2. `process_data.py` filters unavailable records and normalises costs, positions, teams, and availability.
3. `fetch_fixtures.py` stores upcoming fixtures and difficulty ratings.
4. `season_planner.py` derives a chip roadmap and writes `macro_plan.json`.
5. `train_model.py` writes `ai_players_multiweek.csv` with projections for gameweeks 1-38.
6. `multi_week_optimizer_mpc.py` validates the inputs and solves the rolling four-gameweek plan.

The committed files in `data/` are inspectable snapshots, not a substitute for refreshing the official API before making a current decision.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

The tests solve the multi-week model on a small synthetic player pool and check that every planned week respects the game rules (15 players with 2/5/5/3 per position, at most three per club, budget, a valid starting eleven), that every transfer is paid for with a free transfer or a hit, and that the model buys a clearly better player when it can. One test guards against an earlier objective bug: when cash in the bank was rewarded too strongly, the model sold good players for cheaper ones of equal value; with nothing better available it must now keep the squad.

The tests use the HiGHS solver when it is installed, because the CBC binary bundled with PuLP does not run on Apple Silicon without Rosetta.

## Engineering Notes

- The projection layer is a transparent, deterministic scoring model rather than a black-box training service. This keeps feature influence and failure modes inspectable.
- The optimisation output is a recommendation, not an automated trade execution mechanism. Transfers must be applied on the official FPL platform.
- API availability, fixture rescheduling, late team news, and player minutes uncertainty can materially change results. Re-run the acquisition and projection stages before each decision.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).