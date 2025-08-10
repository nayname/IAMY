# Backend Automation with Guardrailed LLMs

This repository contains an open-source backend system leveraging **Large Language Models (LLMs)** within a **Model Context Protocol (MCP)** framework to transform user intents, expressed in natural language, into **validated, reproducible workflows** (“recipes”).

Originally designed for specialized domains, the system now targets **broad backend automation** —  like **DevOps**, **Object storage**, **ETL pipelines** etc.

> **Our Goal: Make AI Agents Predictable for Backend Tasks 🤖**
> LLMs are powerful but often produce hallucinations, unsafe outputs, or actions that are out of scope. This project solves that problem by integrating 
> strict guardrails, precise semantic classification, and vetted workflows to ensure every output is correct, safe, and repeatable.

---
## The Problem 😟
LLMs are like eager assistants who will confidently lead you down the wrong path. Their tendency to be inaccurate or unreliable is a major roadblock for complex tasks , especially when working with specialized tech stacks. Without a system for preconditions and validation, their outputs are annoying at best and dangerous at worst. This lack of guardrails often leads to incorrect execution, false information, or out-of-scope actions

---
## The Solution ✅
LLMs become reliable when integrated within a structured system of guardrails. Our solution is a validation layer built into an MCP server that uses predefined and vetted "recipes".

Here’s how it works:

1. The system **Parses** a user's request with a non-LLM Intent Classifier.
2. **Selects** a vetted “recipe” that matches the user's intent.
3. **Provides** the LLM with the exact instructions, resources, and constraints from that recipe.
4. Finally, **Validates** the LLM's output against the recipe's guardrails before execution.

This reduces the risk of incorrect, incomplete, or unsafe results — a key requirement for automation in production environments.
We continuously evaluate and improve our library of high-quality recipes through both automated processes and human refinement.

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
Domain Coverage – Accuracy depends on available recipes. Users can add new resipes.

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
