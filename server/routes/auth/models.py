from pydantic import BaseModel


class LoginBody(BaseModel):
    login: str  # Either username or email
    password: str

# TODO: Implement a more secure password validation mechanism
class RegisterBody(BaseModel):
    username: str
    email: str
    password: str
