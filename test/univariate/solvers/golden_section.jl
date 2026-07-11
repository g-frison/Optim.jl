@testset "Golden Section" begin
    for (name, prob) in OptimTestProblems.UnivariateProblems.examples
        for T in (Float64, BigFloat)
            results = optimize(
                prob.f,
                convert(Array{T}, prob.bounds)...,
                method = GoldenSection(),
            )

            @test Optim_gf.converged(results)
            @test norm(Optim_gf.minimizer(results) .- prob.minimizers) < 1e-7
        end
    end

    ## corner cases - empty and zero-width brackets
    result = optimize(sqrt, 4.0, 4.0, method = GoldenSection())
    @test Optim_gf.converged(result)
    @test Optim_gf.minimizer(result) == 4.0
    @test Optim_gf.minimum(result) == 2.0
    @test_throws ErrorException optimize(identity, 2.0, 1.0, GoldenSection())

    ## time limit
    function slow_obj(x)
        sleep(0.05)
        return sin(x)
    end
    result = optimize(x -> slow_obj(x), 0, 2π, GoldenSection(); time_limit = 0.01)
    @test result.f_calls == 1

    result = Optim_gf.optimize(
        x -> sin(x),
        0,
        2π,
        Optim_gf.GoldenSection();
        abs_tol = 1e-4,
        store_trace = false,
        show_trace = true,
        iterations = 2,
    )
end
