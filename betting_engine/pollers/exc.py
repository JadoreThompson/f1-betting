class APIError(Exception):
    """Error fetching response from external API."""
    pass

class TargetNotFound(Exception):
    """Next round in season not found"""
    def __init__(self) -> None:
        super().__init__("No target round was found. Season possibly over.")
        

class SeasonOver(Exception):
    def __init__(self) -> None:
        super().__init__("Couldn't find the date for the next grand prix. Season is over.")