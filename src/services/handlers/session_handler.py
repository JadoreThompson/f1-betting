from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..session_poller import SessionPoller


class SessionHandler(ABC):
    """
    Defines the strategy for polling, parsing, and persisting a specific F1 session type.
    This interface is the core of the Strategy Pattern, allowing the main poller
    to handle different session types uniformly.
    """

    @property
    @abstractmethod
    def api_lookup_key(self) -> str:
        """The key in the API response JSON where the results are located."""
        pass

    @property
    @abstractmethod
    def fetch_path(self) -> str:
        """The URL path template for the API endpoint."""
        pass

    @abstractmethod
    def get_target_event_time(
        self, schedule: list[dict[str, Any]]
    ) -> Optional[tuple[int, datetime]]:
        """Finds the next upcoming session of this type in the schedule."""
        pass

    @abstractmethod
    async def process(
        self,
        poller_cls: "SessionPoller",
        raw_results: list[dict[str, Any]],
        year: int,
        round_: int,
        circuit_id: int,
        circuit_ref: str,
    ) -> None:
        """
        Parses the raw data, persists it to the database, and runs any
        necessary post-processing hooks like updating standings or triggering predictions.
        """
        pass
