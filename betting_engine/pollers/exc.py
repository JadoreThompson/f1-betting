class APIError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class TargetNotFound(Exception):
    def __init__(self) -> None:
        super().__init__("No target round was found. Season possibly over.")
        

class SeasonOver(Exception):
    def __init__(self) -> None:
        super().__init__("Couldn't find the date for the next grand prix. Season is over.")