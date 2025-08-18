import json

# from create import generate_code
from lib.classifier import classifier_condition, classify
# from lib.json_gen import generate_json
# from iamy.docsrs_client import call_docsrs

from langgraph.graph import END, START, StateGraph

from lib.docsrs import call_docsrs
from lib.pp import Pipeline
from lib.states import OverallState, InputState, OutputState




class Agent:
    def __init__(self, query, label):
        # LLM
        # self.p = Pipeline()

        self.graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

        self.graph.add_node("classifier", classify)
        self.graph.add_node("contract", self.generate_contract_answer)
        self.graph.add_node("no_contract", self.generate_final_answer)
        # self.graph.add_node("JSONGenerator", self.json_generator_node)
        # self.graph.add_node("DocsrsCaller", self.docsrs_node)

        # Edges/Transitions
        self.graph.add_edge(START, "classifier")
        self.graph.add_conditional_edges(
            "classifier",
            classifier_condition,
        )
        self.graph.add_edge("contract", END)
        self.graph.add_edge("no_contract", END)

        self.graph.compile().invoke({"question": query, "label": label})
        # langgraph.add_node(guardrails)
        # langgraph.add_node(generate_cypher)
        # langgraph.add_node(validate_cypher)
        # langgraph.add_node(correct_cypher)
        # langgraph.add_node(execute_cypher)
        # langgraph.add_node(generate_final_answer)
        #
        # langgraph.add_edge(START, "guardrails")
        # langgraph.add_conditional_edges(
        #     "guardrails",
        #     guardrails_condition,
        # )
        # langgraph.add_edge("generate_cypher", "validate_cypher")
        # langgraph.add_conditional_edges(
        #     "validate_cypher",
        #     validate_cypher_condition,
        # )
        # langgraph.add_edge("execute_cypher", "generate_final_answer")
        # langgraph.add_edge("correct_cypher", "validate_cypher")
        # langgraph.add_edge("generate_final_answer", END)
        #
        # langgraph = langgraph.compile()
        #
        # # View
        # display(Image(langgraph.get_graph().draw_mermaid_png()))

    def generate_final_answer(self, state: InputState)  -> OutputState:
        print("END")
        return {"answer":"END"}

    def generate_contract_answer(self, state: InputState)  -> OutputState:
        print("ITS A CONTRACT")
        return {"answer":"ITS A CONTRACT"}

    # def classifier_node(self):
    #     result = self.classifier.classify_intent()
    #     # Return structure compatible with LangGraph
    #     return {"intent": result}
    #
    # def json_generator_node(self):
    #     f = open('generated_map.json')
    #     map = json.load(f)
    #     die
    #
    #     return generate_code(self.query, map, self.classifier, self.p)

    def docsrs_node(self):
        # Example: make HTTP call to docsrs
        return call_docsrs(self.query)


# Usage
# def main_flow(user_input):
#     graph = Graph()
#     result = graph.run({"user_input": user_input})
#     return result
f = open('queries.json')
queries = json.load(f)

Agent(queries['nft_marketplace'][0], 'nft_marketplace')
Agent(queries['crowdfund'][0], 'crowdfund')
Agent(queries['cw20_exchange'][0], 'cw20_exchange')
