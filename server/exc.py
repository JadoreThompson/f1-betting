class JWTError(Exception):
    """Custom exception for JWT errors."""

    def __init__(self, message: str | None = None):
        message = message or "Error authenticating user"
        super().__init__(message)
        self.message = message