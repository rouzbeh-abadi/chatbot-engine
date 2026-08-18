from dataclasses import dataclass

from pydantic import BaseModel


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float | None = None

@dataclass
class LLMResponse[T]:
    data: T
    usage: TokenUsage | None = None