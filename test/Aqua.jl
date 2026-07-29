using Aqua
using OnlineML

@testset "Aqua.jl" begin
    Aqua.test_all(
        OnlineML;
        ambiguities = (broken = false,),
        deps_compat = (check_extras = true, check_weakdeps = true),
        stale_deps = (ignore = [:Documenter, :Plots],),
        # Disable undefined_exports check - exporting submodules (Pipeline, Linear, etc.)
        # causes false positives on some Julia versions/platforms (Julia 1.10 Windows)
        undefined_exports = false,
        piracies = true,
        persistent_tasks = true,
    )
end
