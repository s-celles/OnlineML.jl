"""
    Optim

Re-exports gradient optimizers from Optimisers.jl for use with online learning models.

# Available Optimizers

- `SGD(η)` - Stochastic Gradient Descent
- `Momentum(η, ρ)` - SGD with momentum
- `Adam(η, β)` - Adaptive Moment Estimation
- `AdaGrad(η)` - Adaptive Gradient
- `RMSProp(η, ρ)` - Root Mean Square Propagation
- `AdamW(η, β, λ)` - Adam with weight decay

# Example

```julia
using OnlineML.Optim

model = LogisticRegression(optimizer=Adam(0.01))
```
"""
module Optim

using Optimisers

# Re-export common optimizers (only those defined in Optimisers.jl)
export Momentum, Adam, AdaGrad, RMSProp, AdamW
export Descent, Nesterov, RAdam, NAdam, AdaMax, OAdam

"""
    default_optimizer()

Return the default optimizer for gradient-based models: Adam(0.001)
"""
default_optimizer() = Adam(0.001)

# Convenience aliases
const SGD = Descent
export SGD

end # module Optim
