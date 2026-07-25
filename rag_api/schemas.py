from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="User question in Persian.",
        min_length=1, # Empty string "" is rejected
        max_length=2000,
    )

class AskResponse(BaseModel):
    Answer: str = Field(
        ...,
        description="HTML answer (RTL, Persian)",
    )
    status: str = Field(
        ...,
        description="ok | fallback | medical_redirect | out_of_scope | empty",
    )


