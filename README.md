# Backend Automation with Guardrailed LLMs

This repository contains an open-source backend system leveraging **Large Language Models (LLMs)** within a **Model Context Protocol (MCP)** framework to transform user intents, expressed in natural language, into **validated, reproducible workflows** (“recipes”).

Originally designed for specialized domains, the system now targets **broad backend automation** —  like **DevOps**, **Object storage**, **ETL pipelines** etc.

> **We help AI agents with predictible backend stuff**  
> LLMs are powerful but prone to hallucinations, unsafe outputs, and off-scope actions.  
> This project integrates **guardrails, semantic classification, and vetted workflows** to ensure outputs are correct, safe, and repeatable.

Problem
LLMs are people pleasers that will confidently lead you
down the wrong path. They are often inaccurate and unreliable, especially for complex tasks. This unpredictability is even more pronounced when building with an exotic
tech stack like CosmWasm. A lack of guardrails leads to
false information, incorrect execution, or out of scope
tasks. Without a system for preconditions and validation,
their outputs are annoying at best and dangerous at worst.
Solution
LLMs can be made more reliable by integrating them
within guardrails. We introduce a validation layer in the
MCP server built on predefined and vetted ”recipes.”
We continuously evaluate and improve these recipes
through a self improving process, including human refinement, to build a library of high quality workflows. These
recipes provide context and specific instructions to the
LLM and help validate LLM outputs. This process prevents
incorrect or unsafe actions from being executed. The ultimate goal is to develop and expand this library of
recipes to the point where LLM agents can be trusted
in production environments.

---

## 📚 Project Overview

1. **Parses** the request using a non-LLM **Intent Classifier**.
2. **Selects** a vetted “recipe” matching the request.
3. **Provides** the LLM with exact instructions, resources, and constraints from that recipe.
4. **Validates** the LLM output against guardrails before execution.

This reduces the risk of incorrect, incomplete, or unsafe results — a key requirement for automation in production environments.

Backend automation often requires translating a user’s high-level request into a precise sequence of technical steps.  
This project addresses that challenge by:

* Accepting a user’s intent in plain language (e.g., “Create a monthly passenger traffic chart from CSV data”).
* Using an **Intent Classifier** to match the request with a vetted recipe.
* Providing the LLM with structured instructions, resources, and validation rules from the recipe.
* Validating the LLM’s output against predefined guardrails before execution.

By combining **structured recipes**, **semantic intent classification**, and **output validation**, the system enables more reliable, safe, and repeatable automation.

---

## ⚙️ Architecture

### MCP Server
Implements the [Model Context Protocol](https://modelcontextprotocol.io/) specification.  
Exposes tools that can be called by compatible clients (e.g., Cursor, Claude Code).

### Library of Recipes
A curated JSON-schema library of tested workflows.  
Each recipe contains:
- **Intent description**
- **Required tools**
- **Step-by-step workflow**
- **Optional context** (docs, boilerplates, data samples)

### Intent Classifier
Matches user input to the correct recipe without calling an LLM.  
Ensures the right plan is selected before LLM involvement.

### Guardrails & Fallback Rules
Validates LLM outputs against the recipe specification.  
Rejects unsafe, incomplete, or out-of-scope results.

---

## 💾 Usage

To run sample generation:

```bash
python create.py
Generated recipes and outputs are stored in /generated.

⚠️ Known Limitations
Domain Coverage – Accuracy depends on available recipes.

LLM Hallucinations – Mitigated by guardrails, but incomplete recipes can still allow errors.

Edge Cases – May require additional validation layers.

🚀 Possible Improvements
Expand recipe coverage for more backend domains.

Automated CI testing of generated outputs.

Metrics & monitoring of recipe performance and error rates.

Community recipe submissions via schema-based PRs.

📜 License
MIT License.
See LICENSE for details.
