# behaviour.py — DANS move classifier + TFT concession tracker + PA awareness
# HERALD HAN 2026 agent

from __future__ import annotations

DANS_TYPES = ("silent", "nice", "concession", "unfortunate", "fortunate", "selfish")
_THRESHOLD = 0.01


def _classify(my_change: float, opp_change: float) -> str:
    my_up = my_change > _THRESHOLD
    my_dn = my_change < -_THRESHOLD
    op_up = opp_change > _THRESHOLD

    if not my_up and not my_dn and not op_up and opp_change >= -_THRESHOLD:
        return "silent"
    if not my_dn and op_up:
        return "nice"
    if my_dn and op_up:
        return "concession"
    if my_dn and not op_up:
        return "unfortunate"
    if my_up and op_up:
        return "fortunate"
    if my_up and not op_up:
        return "selfish"
    return "silent"


class BehaviourTracker:
    """
    Tracks opponent and agent DANS move types, TFT concession ratio, and
    PA (partner awareness) coefficient.

    All moves are classified from the perspective of the party making the move
    — so for record_opponent, arguments are swapped to opponent's perspective.
    """

    def __init__(self) -> None:
        self._opp_moves: list[str] = []
        self._agent_moves: list[str] = []

        # PA awareness coefficient
        self._pa_num: int = 0
        self._pa_den: int = 0

        # TFT concession tracking (in utility units)
        self._opp_concession_amounts: list[float] = []
        self._agent_concession_amounts: list[float] = []
        self._total_opp_concession: float = 0.0
        self._total_agent_concession: float = 0.0

    def record_opponent(
        self,
        agent_util_change: float,
        opp_util_change: float,
        opp_concession_amount: float = 0.0,
    ) -> str:
        """
        Record the opponent's move.

        Arguments use NEXUS perspective:
          agent_util_change = change in HERALD's utility from opponent's offer
          opp_util_change   = estimated change in opponent's own utility

        For DANS classification we swap to opponent's perspective:
          opponent's "my_change" = opp_util_change
          opponent's "opp_change" = agent_util_change
        """
        move = _classify(opp_util_change, agent_util_change)
        self._opp_moves.append(move)
        if opp_concession_amount > 0.0:
            self._opp_concession_amounts.append(opp_concession_amount)
            self._total_opp_concession += opp_concession_amount
        return move

    def record_agent(
        self,
        agent_util_change: float,
        opp_util_change: float,
        agent_concession_amount: float = 0.0,
    ) -> str:
        """
        Record HERALD's own move and update PA coefficient.
        """
        move = _classify(agent_util_change, opp_util_change)

        if self._agent_moves:
            prev = self._agent_moves[-1]
            if move != prev:
                self._pa_den += 1
                if (
                    len(self._opp_moves) >= 2
                    and self._opp_moves[-1] != self._opp_moves[-2]
                ):
                    self._pa_num += 1

        self._agent_moves.append(move)
        if agent_concession_amount > 0.0:
            self._agent_concession_amounts.append(agent_concession_amount)
            self._total_agent_concession += agent_concession_amount
        return move

    @property
    def pa(self) -> float:
        """PA awareness coefficient — fraction of HERALD's move changes the
        opponent also changed. Default 0.5 until enough data."""
        if self._pa_den == 0:
            return 0.5
        return min(1.0, max(0.0, self._pa_num / self._pa_den))

    @property
    def agent_total_concession(self) -> float:
        return self._total_agent_concession

    @property
    def opp_total_concession(self) -> float:
        return self._total_opp_concession

    @property
    def tft_ratio(self) -> float:
        """
        Opponent total concession / agent total concession.
        > 1.0: opponent is conceding more → HERALD can hold firm.
        < 1.0: HERALD conceding more → consider matching.
        """
        if self._total_agent_concession < 0.001:
            return 1.0
        return self._total_opp_concession / (self._total_agent_concession + 1e-9)

    def is_opponent_conceding(self, window: int = 5) -> bool:
        """True if >50% of the last `window` opponent moves are concession/nice."""
        recent = self._opp_moves[-window:]
        if not recent:
            return False
        n = sum(1 for m in recent if m in ("concession", "nice"))
        return n / len(recent) > 0.5

    def is_opponent_stuck(self, window: int = 6) -> bool:
        """True if >60% of the last `window` opponent moves are silent/selfish —
        opponent is not moving toward HERALD."""
        recent = self._opp_moves[-window:]
        if len(recent) < window:
            return False
        n = sum(1 for m in recent if m in ("silent", "selfish"))
        return n / len(recent) > 0.6

    @property
    def opponent_moves(self) -> list[str]:
        return list(self._opp_moves)

    @property
    def agent_moves(self) -> list[str]:
        return list(self._agent_moves)
