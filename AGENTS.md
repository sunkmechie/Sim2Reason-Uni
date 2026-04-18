# AGENTS.md

FOR THE CODING AGENTS THAT ARE ASSISTING / MIGHT ASSIST WITH THIS WORKFLOW.


This repository is being adapted for backend-agnostic physics simulation research.
The goal is to decouple scene semantics from MuJoCo-specific implementation details
so the same scene IR can target multiple simulators, including MuJoCo and Genesis.

## Operating Principles

1. Preserve scientific intent.
   - Prefer changes that improve backend portability, reproducibility, and evaluation quality.
   - Do not optimize for convenience if it weakens the research story.

2. Separate intent from execution.
   - Scene semantics, connectivity, physics parameters, and labels belong in an IR.
   - Backend code should only compile IR into simulator-specific objects and runtime state.

3. Keep MuJoCo as one backend, not the source of truth.
   - Avoid introducing new MuJoCo-only concepts into shared domain code.
   - If a concept is backend-specific, isolate it behind an adapter or an explicit backend tag.

4. Prefer additive, reversible changes.
   - Refactors should preserve existing outputs until parity is proven.
   - Introduce compatibility layers before deleting legacy paths.

5. Protect the research branch.
   - Do not make unrelated cleanup edits while working on backend abstraction.
   - Keep experiments, benchmarks, and design notes in `experiments/`. (ALWAYS MAKE SURE THIS FOLDER IS IN .gitignore and NEVER COMMITED TO GIT)

## Coding Rules

1. Do not run code unless explicitly requested or unless the task depends on it.
   - When environment setup is missing, inspect and reason from source only.

2. Do not use destructive git or filesystem commands.
   - Never use `git reset --hard`, `git checkout --`, or unapproved deletions.

3. Prefer small, reviewable patches.
   - One logical change per patch.
   - Keep API shifts staged and documented.

4. Preserve backward compatibility where practical.
   - Existing scene YAMLs, entity names, and outputs should keep working during migration.

5. Avoid hidden coupling.
   - Do not let parsers, emitters, recorders, and evaluators infer meaning from compiled simulator internals if the same meaning can be represented in IR.

6. Add comments only where the code is non-obvious.
   - Comments should explain intent, not restate code.

7. Use ASCII by default.
   - Introduce non-ASCII only if it already exists or is clearly justified.

## Backend-Agnostic Design Rules

1. IR must be stable and explicit.
   - Every entity, body, site, joint, tendon, sensor, actuator, and contact relationship should have a stable identity.

2. Backend emitters must be pure translators.
   - They should not invent new scene semantics.
   - They may add backend-required metadata, but only through explicit extension fields.

3. Runtime adapters must own simulator-specific logic.
   - XML generation, model compilation, stepping, metric extraction, and rendering should be backend-scoped.

4. Shared logic must never depend on simulator internals.
   - Anything that needs `MjModel`, `MjData`, or simulator-specific IDs belongs behind the MuJoCo adapter.

5. Validate cross-backend equivalence.
   - For each representative scene family, compare IR, emitted scene, runtime observables, and generated QA artifacts.

## Research Discipline

1. Keep a clear trail of decisions.
   - Record architecture choices, rejected alternatives, and benchmark assumptions in `experiments/roadmap.md` and `experiments/plan.md`.

2. Treat performance claims as hypotheses until measured.
   - Speedup, fidelity, and quality improvements must be benchmarked on the same scene families.

3. Prioritize reproducibility.
   - Fix seeds, scene families, evaluation sets, and metric definitions before making performance claims.

4. Optimize for a publishable contribution.
   - The target outcome is a clean scientific story, not just a working port.

## Workflow Expectations

1. Read before editing.
2. Refactor toward a backend-neutral IR first.
3. Keep MuJoCo parity tests passing while adding new backends.
4. Document the roadmap before expanding implementation scope.

