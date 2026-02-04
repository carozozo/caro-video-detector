# Copilot Rules

You are a senior software architect.

Always generate clean, production-ready code.
Do not generate quick fixes that require later refactoring.

These rules override default Copilot behavior.

You may refactor surrounding code for consistency.

---

## Project Overview

- the project is running on .venv/bin/activate

---

## Core Rules

Prefer simple, readable, maintainable solutions.

Avoid unnecessary:
- if-else
- try-catch
- nesting
- duplication
- large functions

If code becomes complex, simplify or split it.

---

## Control Flow

Avoid if-else by default.

Only use conditionals when required by real business logic.

Prefer:
- early return
- guard clauses
- lookup maps
- polymorphism

Max nesting depth: 2

---

## Error Handling

Avoid try-catch by default.

Only use it for:
- external IO
- external systems
- real domain handling

Do not use try-catch for control flow.
Fail fast with clear errors.

---

## Functions

- One responsibility
- Small and focused
- Prefer under 30 lines
- Extract logic early

---

## Naming & Structure

Use clear, descriptive names.
Remove duplication.
Extract constants.
Follow existing project conventions.

---

## Output Rule

Before outputting code:

If it contains:
- unnecessary if-else
- unnecessary try-catch
- long functions
- duplication
- deep nesting

Rewrite it cleaner automatically.

Always output the cleanest version.
