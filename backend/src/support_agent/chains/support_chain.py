from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from support_agent.llm import create_llm
from support_agent.models.support import SupportResponse

SYSTEM_PROMPT = """
You are a customer support assistant.

Your job is to help customers with questions about the product,
subscriptions, orders, billing, and troubleshooting.

Be clear, concise, and helpful.
Do not invent information you do not know.
"""


def create_support_chain() -> Runnable:
    """Create the basic customer support conversation chain."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{message}"),
        ]
    )

    llm = create_llm()

    structured_llm = llm.with_structured_output(
        SupportResponse,
        include_raw=True,
    )

    return prompt | structured_llm