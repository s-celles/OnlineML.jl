using Aqua
using OnlineML

@testset "Aqua.jl" begin
    Aqua.test_all(
        OnlineML;
        ambiguities = (broken = false,),
        deps_compat = (check_extras = true, check_weakdeps = true),
        stale_deps = (ignore = [:Documenter, :Plots],),
        piracies = true,
        persistent_tasks = true,
    )
end
