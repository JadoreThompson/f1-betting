from typing import Optional
from pydantic import BaseModel, field_validator


class LoginBody(BaseModel):
    login: str  # Either username or email
    password: str

# TODO: Implement a more secure password validation mechanism
class RegisterBody(BaseModel):
    username: str
    email: str
    password: str

    @classmethod
    def _validate_field(cls, value: Optional[str], field_name: str) -> str:
        if not value:
            raise ValueError(f"{field_name.capitalize()} cannot be empty.")
        return value

    @field_validator("username")
    def validate_username(cls, value: Optional[str]) -> str:
        return cls._validate_field(value, "username")

    @field_validator("email")
    def validate_email(cls, value: Optional[str]) -> str:
        return cls._validate_field(value, "email")

    @field_validator("password")
    def validate_password(cls, value: Optional[str]) -> str:
        return cls._validate_field(value, "password")

