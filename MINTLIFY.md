# Mintlify Integration Notes (External, Opt-In)

This document explains how IAMY can be integrated into a **Mintlify-style MDX documentation site** as an **external execution layer**, without requiring changes to Mintlify core.

The goal is to explore **executable documentation** while keeping **execution responsibility outside the docs platform**.

---

## What problem this explores

Documentation often explains *how* to do things, but users still need to:
- translate intent into commands / API calls
- reason about side effects
- manually execute steps

IAMY explores whether docs can support a safer pattern:

> **Intent → deterministic plan → preview → explicit confirmation → execution → logs**

This repo treats Mintlify as a **host UI**, not an execution engine.

---

## Design principles (important)

This integration is intentionally conservative:

- **Preview-first**: nothing executes without a visible plan
- **Explicit confirmation**: no implicit or automatic execution
- **External execution**: Mintlify does not run or authorize actions
- **No secrets in UI**: execution credentials live server-side
- **Clear responsibility boundary**: IAMY owns execution semantics

If any of these principles are unacceptable, the feature should remain external.

---

## Minimal integration surface

### MDX usage (conceptual)

```mdx
import { IamyExecWidget } from "@/components/IamyExecWidget";

<IamyExecWidget
  endpoint="https://YOUR_IAMY_BACKEND"
  mode="preview"     // "mock" | "preview" | "execute"
  suggestions={[
    "Generate the CLI command to deploy the contract",
    "Show the API request needed to list resources",
  ]}
/>
````

### Widget responsibilities

The embedded widget is responsible for:

1. Collecting user intent (free text)
2. Calling IAMY `/api/plan`
3. Rendering the structured plan
4. Requiring an explicit **Confirm** action
5. Calling `/api/execute` only after confirmation
6. Displaying logs / results

The widget **must not**:

* auto-run on load
* store secrets
* infer confirmation from user text

---

## Backend responsibilities (IAMY)

The IAMY backend is responsible for:

* Generating **deterministic Execution Plans**
* Validating plans before execution
* Enforcing allowlists / policies
* Running execution in controlled environments
* Returning structured logs and receipts

Mintlify is never responsible for:

* authorization
* side effects
* execution safety
* audit storage

---

## Suggested repo layout

A minimal structure for a Mintlify demo looks like:

```
packages/mintlify-widget/
  src/
    IamyExecWidget.tsx
    types.ts

examples/mintlify-site/
  docs/
    exec-demo.mdx
  components/
    IamyExecWidget.tsx
  mint.json
```

The demo site should be runnable locally without Mintlify privileges.

---

## Execution modes

To reduce risk, the widget should expose clear modes:

* `mock`
  Frontend-only demo, no backend execution

* `preview` (recommended default)
  Generates and displays plans only

* `execute`
  Allows confirmed execution via backend

Docs should default to **preview**.

---

## Why this should remain external (initially)

From a platform perspective:

* Execution introduces **liability**
* Deterministic planning is **non-trivial**
* Audit / replay requirements are ongoing
* Policies vary per user / org / environment

IAMY is designed to absorb this complexity so that:

* Mintlify focuses on docs UX
* Execution evolves independently
* Users opt in deliberately

---

## Questions for Mintlify maintainers

This integration is exploratory. Useful questions include:

* Would executable docs ever be in scope for Mintlify core?
* If not, what extension points are preferred?
* Should execution always remain external?
* What safety guarantees would you expect from third-party execution layers?

---

## What this is *not*

* Not a request to merge into Mintlify core
* Not a generic AI chat widget
* Not an autonomous agent
* Not a replacement for documentation clarity

It is an exploration of **responsible execution inside documentation**.

---

## Status

Current state:

* External widget concept
* Deterministic plan → confirm → execute flow
* Reference API contract

Future exploration depends on maintainer feedback and real usage.
