class JWTError(Exception):
    """Custom exception for JWT errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message