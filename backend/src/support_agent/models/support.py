from pydantic import BaseModel, Field


class SupportResponse(BaseModel):
    answer: str = Field(
        description="The response to the customer's message."
    )