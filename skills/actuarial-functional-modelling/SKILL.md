---
name: "actuarial-functional-modelling"
description: "Use when building, refactoring or reviewing a Python actuarial projection model (pricing, valuation, reinsurance) to enforce pure-function, array-based design."
---

# Actuarial functional modelling

Models are built from pure functions over arrays. No cell graphs, no hidden state, no implicit caching. Write and verify in NumPy first; the design must port to other backends (Numba, JAX) without restructuring.

## Core rules

1. **Pure functions.** Output depends only on arguments. Calculation functions do not read globals, module-level tables or files, and do not cache.
2. **Explicit state.** Everything carried between time steps lives in one immutable state record (`NamedTuple` or frozen `dataclass`).
3. **Step function signature:** `step(state, inputs_t, params) -> (state_next, outputs_t)`. A step never indexes `t-1`; the previous value is in `state`.
4. **One driver owns the time loop.** It iterates `step` over `t` and stacks `outputs_t`. Product logic never appears in the driver.
5. **Vectorise over policies and scenarios; loop over time.** Array axes are `(n_scen, n_pol)`. No per-policy Python loops.
6. **Data in, data out.** Model points, assumptions and scenarios are loaded and validated once at the I/O boundary and passed in as arrays. The calculation layer never touches pandas or files.
7. **One function per actuarial quantity.** Name the function after the quantity: `survival_factor`, `credited_rate`, `surrender_charge`, `modco_adjustment`.

## Layering

```
io            load + validate inputs -> typed records
assumptions   lookups -> arrays aligned to the projection grid (precomputed, not looked up per step)
liability     step + driver -> gross liability cashflows and reserves
treaty        gross cashflows + treaty terms -> ceded, retained
assets        investment income, discounting
results       PVs, IRR, profit signature, aggregation
```

Each layer is a function of the previous layer's outputs. The liability layer never knows the treaty exists.

## Conventions

State these once in the package's top-level module docstring and follow them everywhere:

- **Period:** monthly unless stated. Convert annual rates to the period rate in one place (`_pa` → `_pm`).
- **Timing suffixes:** `_bop`, `_mid`, `_eop` on every time-located quantity.
- **Decrement order:** written in one function (e.g. deaths mid-period, then lapses at period end) and reused.
- **Sign convention:** inflows to the modelled entity positive. The treaty layer states which entity it is modelling (cedant or reinsurer).
- **Units:** per-policy amounts unless suffixed `_total`.
- **Precision:** float64 throughout.
- **Parameters:** no literals in calculation code. Parameters live in a frozen `params` record with documented defaults and units.
- **Shapes:** every array argument's shape is in the docstring, e.g. `(n_scen, n_pol)`.

## Treaty layer

- Signature: `treaty_fn(gross: CashflowRecord, terms: TreatyTerms) -> (ceded, retained)`.
- Coinsurance, modco, funds withheld, experience refund and recapture are separate functions that compose.
- Assert `retained + ceded == gross` for every cashflow line on every run.

## Testing

- **Hand-calculation unit tests.** Each function gets a 2–3 period example worked by hand, with the working in the test docstring.
- **Invariant tests:**
  - gross = retained + ceded
  - in-force count non-increasing with no new business
  - zero decrements → account value equals deterministic roll-up
  - earned rate = credited rate and zero expenses → zero spread profit
  - PV at 0% discount = undiscounted sum
- **Reference implementation.** A slow single-policy loop written for readability. The vectorised model must match it to 1e-10.
- **Regression snapshots.** Freeze PV profit, IRR and reserves at t = 0, 1, 10 for sample model points. Any change to a snapshot is explained in the commit message.

## Portability rules (backend-agnostic)

Write step functions so they can move to a compiled backend unchanged:

- Never mutate input arrays; build and return new ones.
- No Python `if` on array values inside a step; use `np.where`.
- No data-dependent output shapes (e.g. boolean-mask filtering) inside a step.
- Optimise only after the regression suite exists.

## Review checklist

- [ ] Does any function read something that is not an argument?
- [ ] Is any assumption looked up inside the time loop that could be precomputed?
- [ ] Is bop/eop timing consistent at every multiplication?
- [ ] Are rate conversion and decrement order each done in exactly one place?
- [ ] Is the sign convention consistent through the treaty layer?
- [ ] Does every new function have a hand-calculation test?
- [ ] Does the vectorised model still match the reference implementation?

## Anti-patterns

- Cell-graph frameworks with implicit memoisation (`pols_if(t)` calling `pols_if(t-1)`): hidden state, recursion limits, hard to vectorise.
- Classes holding mutable projection state.
- pandas operations or dataframe merges inside the time loop.
- Product logic copied between models instead of extracted into a shared function.
- Treaty terms leaking into liability code.
