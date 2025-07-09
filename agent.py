import json

from create import generate_code
from lib.classifier import Classifier
# from lib.json_gen import generate_json
# from iamy.docsrs_client import call_docsrs

from langgraph.graph import Graph

from lib.docsrs import call_docsrs
from lib.pp import Pipeline


class Agent:
    def __init__(self, query):
        self.query = query
        self.classifier = Classifier(query)

        # LLM
        self.p = Pipeline()

        self.graph = Graph()

        self.graph.add_node("Classifier", self.classifier_node)
        self.graph.add_node("JSONGenerator", self.json_generator_node)
        self.graph.add_node("DocsrsCaller", self.docsrs_node)

        # Edges/Transitions
        self.graph.add_edge("start", "Classifier")
        self.graph.add_edge("Classifier", {
            "generate_json": "JSONGenerator",
            "use_docsrs": "DocsrsCaller"
        })
        self.graph.add_edge("JSONGenerator", "end")
        self.graph.add_edge("DocsrsCaller", "end")

    def classifier_node(self):
        result = self.classifier.classify_intent()
        # Return structure compatible with LangGraph
        return {"intent": result}

    def json_generator_node(self):
        f = open('generated_map.json')
        map = json.load(f)
        die

        return generate_code(self.query, map, self.classifier, self.p)

    def docsrs_node(self):
        # Example: make HTTP call to docsrs
        return call_docsrs(self.query)


# Usage
# def main_flow(user_input):
#     graph = Graph()
#     result = graph.run({"user_input": user_input})
#     return result
Agent("test")
