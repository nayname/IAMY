# IAMY - Deterministic Execution Layer for Software Interfaces

IAMY is an **execution substrate** that turns **natural-language intent** into a **deterministic execution plan**, with clear preview, confirmation, and guarded execution. It is designed as a **backend execution service** for interfaces such as documentation platforms, dashboards, and AI agents — not as an autonomous agent.

This repo includes:
- A **reference API backend** (`/plan`, `/execute` semantics)
- A **Mintlify integration demo**
- Adapter examples for other surfaces (dashboards, agents)
- JSON schemas for structured plans and results

> Intent → **deterministic Execution Plan** → preview → confirm → execute → logs & audit

---

## 🔎 What IAMY Solves

Modern interfaces let users **read about tasks** but still require them to manually translate intent into actions (CLI, API calls, transactions). This leads to friction, errors, and poor UX.

IAMY enables execution sheets that are:
- **Deterministic** — no hallucination
- **Previewable** — plan first, execute later
- **Guarded** — explicit user confirmation
- **Auditable** — structured results & logs
- **Surface-agnostic** — works with docs, dashboards, agents

---

## 🚀 Quickstart — Mintlify Example (3 min)

> 🍃 This example embeds an execution widget into a Mintlify MDX doc.  
> You don’t need to modify Mintlify core — this is external and opt-in.

1. Clone the repo  
```bash
git clone https://github.com/nayname/IAMY.git
cd IAMY/examples/mintlify-site
