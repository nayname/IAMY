# Mintlify Integration Notes (External, Opt-In)

This document describes how IAMY can be integrated into a **Mintlify-style MDX documentation site** as an **external execution layer**, without requiring changes to Mintlify core.

The goal is to explore **executable documentation** while keeping **execution responsibility, safety, and policy enforcement outside the docs platform**.

---

## What problem this explores

Documentation traditionally explains *what to do*, but users still need to:

- translate intent into commands or API calls
- reason about side effects
- manually execute steps in the correct order

As documentation becomes more interactive and agent-assisted, a new question emerges:

> How can docs enable **user-initiated actions** without introducing unsafe or irreversible behavior?

IAMY explores a conservative execution pattern:

> **Intent → deterministic execution plan → preview → explicit confirmation → execution → audit**

In this model, Mintlify acts as a **host UI**, not an execution engine.

---

## Design principles (important)

This integration is intentionally conservative and safety-first:

- **Preview-first**  
  Nothing executes without a visible, reviewable plan.

- **Explicit confirmation**  
  No implicit, inferred, or automatic execution.

- **External execution responsibility**  
  Mintlify does not execute, authorize, or validate actions.

- **No secrets in the UI**  
  Credentials and permissions live server-side only.

- **Clear responsibility boundary**  
  IAMY owns execution semantics, guardrails, and auditability.

If these constraints are unacceptable, execution should remain out of scope for documentation.

---

## Minimal integration surface

### MDX usage (conceptual)

```mdx
import { IamyExecWidget } from "@/components/IamyExecWidget";

...
