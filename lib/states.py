from operator import add
from typing import Annotated, List

from typing_extensions import TypedDict


class InputState(TypedDict):
    question: str


class OverallState(TypedDict):
    question: str
    next_action: str
    # cypher_statement: str
    # cypher_errors: List[str]
    # database_records: List[dict]
    # steps: Annotated[List[str], add]


class OutputState(TypedDict):
    answer: str
    # steps: List[str]
    # cypher_statement: str