# Upstream bugs and API discrepancies

This file records issues observed in upstream dependencies while developing
OnlineML.jl. Entries distinguish verified upstream behavior from OnlineML defects
and document the local workaround.

## LearnAPI: `clone` calls the advertised keyword constructor positionally

- **Upstream package:** LearnAPI.jl 2.0.1
- **Observed:** 2026-07-29
- **Status:** Reproduced; not yet reported upstream
- **Upstream symbols:** `LearnAPI.clone`, `LearnAPI.constructor`
- **Upstream files:** `src/clone.jl`, `src/traits.jl`

The `constructor` trait documentation requires a keyword constructor and demonstrates
reconstruction with:

```julia
LearnAPI.constructor(learner)(; named_properties...)
```

However, `LearnAPI.clone` collects the learner properties and passes them as positional
arguments:

```julia
LearnAPI.constructor(learner)(NamedTuple{names}(new_values)...)
```

### Minimal reproducer

```julia
using LearnAPI

Base.@kwdef struct DemoLearner
    rate::Float64 = 0.1
end

LearnAPI.constructor(::DemoLearner) = DemoLearner

LearnAPI.clone(DemoLearner(); rate=0.2)
```

Expected: `DemoLearner(0.2)` is returned through the documented keyword-constructor
contract.

Actual: `clone` calls `DemoLearner(0.2)` positionally. A learner that intentionally
provides only the documented keyword constructor raises `MethodError`.

### OnlineML impact and workaround

LearnAPI learner configurations with properties, starting with
`LogisticRegressionLearner`, need an additional positional constructor solely for
compatibility with `LearnAPI.clone`. OnlineML does not overload `LearnAPI.clone`.

The workaround should remain until LearnAPI either:

1. changes `clone` to splat the reconstructed named tuple as keywords; or
2. changes and documents the trait contract as requiring a positional constructor.

Changing `clone` upstream may affect existing learners that implemented only a
positional constructor, so an upstream compatibility transition may be necessary.
