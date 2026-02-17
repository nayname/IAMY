
# IAMY - Executable Docs with Deterministic & Secure Execution
*Don’t enforce logic in prompts. Enforce it in infrastructure.*

IAMY is an **external execution layer** that makes LLM-driven actions explicit, previewable, and policy-bound. It turns natural-language intent into a **deterministic execution plan**, which can be inspected, validated, and confirmed before any real action is executed.

IAMY is designed as a **backend execution service** that sits between LLM systems and real infrastructure.

> Intent → explicit execution plan → validate → execute → logs & audit

---

## Status

Active development happening at:
- [Execution Plan PR](https://github.com/nayname/openclaw-secure-stack/pull/1) — concrete implementation
- [OpenClaw governance plugin](https://github.com/yi-john-huang/openclaw-secure-stack) — execution plan hooks for OpenClaw

---

## 🔎 The Problem: The Leap of Trust in LLM-Driven Execution

As soon as real actions can be proposed by an LLM — whether in automated pipelines, internal tools, or interactive interfaces — users face a leap-of-trust problem. Once execution is mediated by a model, users need strong guarantees about:

* what will happen before anything runs
* explicit confirmation and responsibility
* predictable, repeatable execution

Today, these questions are often resolved implicitly at runtime by the model itself.

IAMY addresses this by introducing a **safety-oriented execution layer** that:

* **Deterministic** — no hidden steps or hallucinated actions
* **Previewable** — execution is proposed before it runs
* **Guarded** — nothing executes without explicit confirmation
* **Auditable** — structured logs and results
* **Surface-agnostic** — works behind any LLM-driven interface

---

## 🧠 Core Concepts

### Intent

A natural-language description of what a user wants to do.

### Execution Plan

A **deterministic, structured plan** that transforms an intent into explicit, reviewable steps.

Unlike prompt-based agent skills, an IAMY Execution Plan is not guidance for the model — it is a concrete, permissioned artifact interpreted and enforced by infrastructure.

Execution plans are not free-form outputs: they are validated against predefined schemas and **preventively evaluated using expert-defined rules and policies** before being shown to the user.

### Guardrails

Guardrails define **what actions are allowed to be proposed and executed**. They encode domain knowledge and safety constraints provided by experts (e.g. read-only limits, parameter bounds, allowed operations, environment restrictions).

Guardrails are enforced **before execution**, ensuring unsafe or out-of-scope actions are never presented for confirmation.

### Preview

Before anything runs, the user sees the full execution plan in a UI and can review every step.

### Execute

After explicit user confirmation, the validated plan runs via backend services or adapters.

### Result & Audit

Structured results and logs suitable for auditing, inspection, and replay.

---

## 🏛️ Philosophy

IAMY is built on a simple premise:  
intelligence can propose actions, but **infrastructure must enforce execution**.

For centuries, humans — already a form of general intelligence — have relied on
signatures, checklists, logs, audits, and separation of duties.
Not because of lack of knowledge, but because **safe execution requires
physical, inspectable constraints**.

AI systems are no different.

Prompt-level instructions and agent “skills” can improve reasoning,
but they cannot guarantee safety, determinism, or accountability once actions
affect real systems—APIs, infrastructure, or financial state.

IAMY externalizes execution from the model.
Instead of trusting the agent to behave correctly, IAMY enforces:

- explicit execution plans
- preview and confirmation
- permissioned actions
- deterministic execution
- logs and auditability

This is not a workaround for weak models.
It is a governance layer for applying intelligence—human or artificial—safely at scale.

---

## 💡 What IAMY Is (and Isn’t)

**IAMY *is***   
✅ A backend execution substrate  
✅ Deterministic plan generation  
✅ Guarded execution modes  
✅ Execution adapters for external surfaces 

**IAMY *is not***   
❌ A general “AI agent”   
❌ A mystery execution layer with hidden steps   
❌ A replacement for user intent confirmation   

---

## 👥 Contributing & Feedback

This project is **open source** and structured to explore execution semantics safely.

We are especially interested in contributions that help:

* refine execution plan schemas
* improve adapter patterns
* add UI integrations without assuming host privileges
* explore enterprise safety modes

---

## 📜 License

This project is licensed under **MIT**. See `LICENSE` for details.

---

## 🧠 Why This Matters

Execution is a **different problem** from reasoning. Interfaces (docs, dashboards, agents) benefit from **deterministic, auditable, and confirmable execution support** — but they shouldn’t own the execution logic. IAMY provides that layer so platforms can focus on experience and users can focus on outcomes.
