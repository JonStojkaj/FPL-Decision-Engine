"""Check that the multi-week MILP respects every game rule on a small synthetic pool."""

import pandas as pd
import pulp
import pytest

from src.multi_week_optimizer_mpc import HORIZON, build_mpc_model

POSITIONS = {"GK": 4, "DEF": 10, "MID": 10, "FWD": 6}
XP_COLUMNS = [f"xp_{week}" for week in range(1, HORIZON + 1)]
# HiGHS runs natively on every platform; the CBC binary bundled with PuLP does
# not run on Apple Silicon without Rosetta.
SOLVER = (
    pulp.HiGHS(msg=False)
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True)
    else pulp.PULP_CBC_CMD(msg=False)
)


def make_pool(xp_by_player=None) -> pd.DataFrame:
    rows = []
    player_id = 1
    for position, count in POSITIONS.items():
        for i in range(count):
            row = {
                "position": position,
                "team_name": f"Team{player_id % 10}",
                "now_cost": 4.5 + (i % 5) * 0.5,
            }
            xp = 2.0 + (i % 5) * 0.5
            for column in XP_COLUMNS:
                row[column] = xp
            rows.append((player_id, row))
            player_id += 1
    players = pd.DataFrame({pid: row for pid, row in rows}).T
    players = players.astype({"now_cost": float, **{c: float for c in XP_COLUMNS}})
    if xp_by_player:
        for pid, xp in xp_by_player.items():
            players.loc[pid, XP_COLUMNS] = xp
    players.attrs["xp_columns"] = XP_COLUMNS
    return players


def solve(players, current=None, bank=0.0, free_transfers=1, scratch=False):
    problem, variables = build_mpc_model(
        players,
        current_player_ids=current or set(),
        manager_info={"bank": bank, "free_transfers": free_transfers},
        scratch=scratch,
    )
    status = problem.solve(SOLVER)
    assert pulp.LpStatus[status] == "Optimal"
    return variables


def chosen(variables, name, week):
    return {pid for pid, by_week in variables[name].items() if by_week[week].value() > 0.5}


def starting_squad(players) -> set[int]:
    squad = set()
    for position, count in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
        squad |= set(players.index[players["position"] == position][:count])
    return squad


def test_every_week_respects_the_game_rules():
    players = make_pool()
    variables = solve(players, scratch=True)
    for week in variables["weeks"]:
        squad = chosen(variables, "squad", week)
        lineup = chosen(variables, "lineup", week)
        positions = players.loc[list(squad), "position"].value_counts().to_dict()
        assert len(squad) == 15
        assert positions == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        assert players.loc[list(squad), "team_name"].value_counts().max() <= 3
        assert players.loc[list(squad), "now_cost"].sum() <= 100.0 + 1e-6
        assert len(lineup) == 11 and lineup <= squad
        lineup_positions = players.loc[list(lineup), "position"].value_counts()
        assert lineup_positions.get("GK", 0) == 1
        assert lineup_positions.get("DEF", 0) >= 3
        assert lineup_positions.get("FWD", 0) >= 1


def test_transfers_are_paid_for_with_free_transfers_or_hits():
    players = make_pool()
    variables = solve(players, current=starting_squad(players), bank=2.0, free_transfers=1)
    for week in variables["weeks"]:
        made = len(chosen(variables, "transfer_in", week))
        free = variables["ft_available"][week].value()
        rolled = variables["rolled_ft"][week].value()
        hits = variables["hits"][week].value()
        assert made == pytest.approx(free - rolled + hits)
        assert variables["itb"][week].value() >= -1e-6


def test_no_pointless_transfers_when_nothing_is_better():
    # Regression test for a misspecified objective: when cash was rewarded too
    # much, the model sold good players for cheaper ones of equal value.
    players = make_pool()
    players[XP_COLUMNS] = 3.0
    current = starting_squad(players)
    variables = solve(players, current=current, bank=0.0, free_transfers=1)
    assert all(not chosen(variables, "transfer_in", week) for week in variables["weeks"])
    assert chosen(variables, "squad", variables["weeks"][-1]) == current


def test_buys_a_clearly_better_player_with_a_free_transfer():
    players = make_pool()
    current = starting_squad(players)
    star = max(pid for pid in players.index if players.loc[pid, "position"] == "MID")
    assert star not in current
    players.loc[star, XP_COLUMNS] = 12.0
    variables = solve(players, current=current, bank=5.0, free_transfers=1)
    assert star in chosen(variables, "transfer_in", 1)
    assert variables["hits"][1].value() == pytest.approx(0)
