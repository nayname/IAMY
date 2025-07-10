import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from lib.states import InputState, OverallState

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from lib.states import InputState, OverallState

contract_types = ["nft_marketplace", "crowdfund", "cw20_exchange", "auction_using_cw20_tokens", "extended_marketplace",
                  "commission_based_sales", "vesting_and_staking"]
classify_query = "Lets pretend that we have an LLM app that generates Andromeda Protocol app contracts" \
                 "using user promtps in natural language. You will be given a user's promt. Based on the context, " \
                 "classify the query to one of the following classes. " \
                 "Classes:".replace("***OPERATIONS***", json.dumps(contract_types))
classify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            classify_query,
        ),
        (
            "human",
            ("{question}"),
        ),
    ]
)

"""
The class categorizes user intents into three groups: smart contract creation, other operations, and miscellaneous.
"""
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class GuardrailsOutput(BaseModel):
    decision: Literal["nft_marketplace", "crowdfund"] = Field(
        description="Decision on whether the question is related to movies"
    )


guardrails_chain = classify_prompt | llm.with_structured_output(GuardrailsOutput)


def classify(state: InputState) -> OverallState:
    """
    Decides if the question is related to movies or not.
    """
    guardrails_output = guardrails_chain.invoke({"question": state.get("question")})
    # database_records = None
    # if guardrails_output.decision == "end":
    #     database_records = "This questions is not about movies or their cast. Therefore I cannot answer this question."
    return {"question": state.get("question"), "next_action": guardrails_output.decision}


def classifier_condition(
        state: OverallState,
) -> Literal["nft_marketplace", "crowdfund"]:
    if state.get("next_action") == "nft_marketplace":
        return "nft_marketplace"
    elif state.get("next_action") == "crowdfund":
        return "crowdfund"
