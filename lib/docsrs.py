import requests

def call_docsrs(query):
    url = "http://localhost:8000/api/endpoint"  # adjust for your docsrs server
    response = requests.post(url, json={"query": query})
    return response.json()