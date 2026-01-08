
# IAMY — Executable Docs with Deterministic & Secure Execution

IAMY is an **external execution layer** that lets users safely perform **user-initiated actions** from documentation platforms, dashboards, and agent interfaces. It turns natural-language intent into a **deterministic execution plan**, with explicit preview, confirmation, and guarded execution.

IAMY is designed as a **backend execution service** that works alongside LLM models.

> Intent → explicit execution plan → preview → confirm → execute → logs & audit

---

## 🔎 The Problem: Executable Docs and the Leap of Trust

Modern documentation lets users **read about tasks**, but as docs become more interactive, they increasingly enable actions to be initiated directly from the interface. In financial or otherwise irreversible environments, this creates friction and risk. This creates a **"leap of trust"** problem. Once actions can be proposed by an LLM inside docs, users need strong guarantees about:

* what will happen before anything runs
* explicit confirmation and responsibility
* predictable, repeatable execution

IAMY addresses this by introducing a **safety-oriented execution layer** that:

* **Deterministic** — no hidden steps or hallucinated actions
* **Previewable** — execution is proposed before it runs
* **Guarded** — nothing executes without explicit confirmation
* **Auditable** — structured logs and results
* **Surface-agnostic** — works with docs, dashboards, and agents

---

## 🚀 Quickstart — Mintlify Example (3 min)

> 🍃 This example embeds an execution widget into a Mintlify MDX doc.  
> You don’t need to modify Mintlify core — this is external and opt-in.

1. Clone the repo  
```bash
git clone https://github.com/nayname/IAMY.git
cd IAMY/examples/mintlify-site
````

2. Install dependencies and start

```bash
pnpm install
pnpm dev
```

3. Open in your browser
   👉 Visit `http://localhost:3000/docs/exec-demo` to interact with the execution widget.


---

## 🧠 Core Concepts

### Intent

A natural-language description of what a user wants to do.

### Execution Plan

A **deterministic, structured plan** that transforms an intent into explicit, reviewable steps.

Execution plans are not free-form outputs: they are validated against predefined schemas and **preventively evaluated using expert-defined rules and policies** before being shown to the user.

### Guardrails

Guardrails define **what is allowed to be proposed and executed**. They encode domain knowledge and safety constraints provided by experts (e.g. read-only limits, parameter bounds, allowed operations, environment restrictions).

Guardrails are enforced **before execution**, ensuring unsafe or out-of-scope actions are never presented for confirmation.

### Preview

Before anything runs, the user sees the full execution plan in a UI and can review every step.

### Execute

After explicit user confirmation, the validated plan runs via backend services or adapters.

### Result & Audit

Structured results and logs suitable for auditing, inspection, and replay.


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
