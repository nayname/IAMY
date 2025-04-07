import hashlib
import json
import os
import time

from openai import OpenAI

client = OpenAI(api_key="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")


class Pipeline:
    embed_model = "text-embedding-ada-002"
    index_name = 'gen-qa-openai-fast'
    open_ai_model = "gpt-4"#"gpt-3.5-turbo-0125"
    open_ai_model_role = 'assistant'


    def get_context_file(self, file_):
        try:
            file = client.files.create(file=open(file_, "rb"), purpose='assistants')
            return file
        except Exception as e:
            print(e)
            return None

    def get_context(self, conversation_history, system_context) -> str:
        conversation_history_clear = [{"role": "user", "content": ""}]
        mess = set()

        # system_context = "Let’s work this out in a step-by-step way to ensure clarity. Welcome to the Training Mode. " \
        #                  "In this mode, you're an assistant to software engineer, helping him to analyze the code. " \
        #                  "Please help user by explaining the main blocks of the code that he provides"
        for convers in conversation_history:
            if not convers['content'] in mess:
                mess.add(convers['content'])
                conversation_history_clear.append(convers)
        print([{"role": "system", "content": system_context}] + conversation_history_clear)

        # Formulate prompt using conversation history
        return [{"role": "system", "content": system_context}] + conversation_history_clear

        # + " [Contexto da conversa] " + ctx


    def ask_gpt(self, messages, model='o1'):
        if not model:
            model = self.open_ai_model

        # config.queries.append({"model": model, "messages": messages})
        print(model)

        if model == "o1-mini" or model == "o1":
            request = {"model": model, "reasoning_effort": "low", "messages": messages, "temperature": None}

            response = client.chat.completions.create(
                model=model,
                # temperature=0.2,
                messages=messages,
                # max_tokens=6000
            )
        elif model == 'o3-mini':
            request = {"model": model, "messages": messages, "temperature":None}
            response = client.chat.completions.create(
                model=model,
                #reasoning_effort = "high",
                # temperature=0.2,
                messages=messages,
                # max_tokens=6000
            )
        elif model == 'gpt-4.5-preview' or model == 'gpt-4o':
            request = {"model": model, "messages": messages, "temperature":None}
            response = client.chat.completions.create(
                model=model,
                #reasoning_effort = "high",
                temperature=0.0,
                seed=123,
                messages=messages,
                # max_tokens=6000
            )
        else:
            request = {"model": model, "messages": messages, "temperature":0.2}
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                seed=123,
                messages=messages,
                # max_tokens=6000
            )
        return request, response
