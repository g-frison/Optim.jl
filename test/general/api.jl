# Test multivariate optimization
@testset "Multivariate API" begin
    # We do not overload `Base.minimum` and `Base.maximum`
    @test Optim_gf.minimum !== minimum
    @test Optim_gf.maximum !== maximum

    rosenbrock = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    f = MVP.objective(rosenbrock)
    g! = MVP.gradient(rosenbrock)
    h! = MVP.hessian(rosenbrock)
    initial_x = rosenbrock.initial_x
    T = eltype(initial_x)
    d1 = OnceDifferentiable(f, initial_x)
    d2 = OnceDifferentiable(f, g!, initial_x)
    d3 = TwiceDifferentiable(f, g!, h!, initial_x)

    Optim_gf.optimize(f, initial_x, BFGS())
    Optim_gf.optimize(f, g!, initial_x, BFGS())
    Optim_gf.optimize(f, g!, h!, initial_x, BFGS())
    Optim_gf.optimize(d2, initial_x, BFGS())
    Optim_gf.optimize(d3, initial_x, BFGS())

    Optim_gf.optimize(f, initial_x, BFGS(), Optim_gf.Options())
    Optim_gf.optimize(f, g!, initial_x, BFGS(), Optim_gf.Options())
    Optim_gf.optimize(f, g!, h!, initial_x, BFGS(), Optim_gf.Options())
    Optim_gf.optimize(d2, initial_x, BFGS(), Optim_gf.Options())
    Optim_gf.optimize(d3, initial_x, BFGS(), Optim_gf.Options())

    Optim_gf.optimize(d1, initial_x, BFGS())
    Optim_gf.optimize(d2, initial_x, BFGS())

    Optim_gf.optimize(d1, initial_x, GradientDescent())
    Optim_gf.optimize(d2, initial_x, GradientDescent())

    Optim_gf.optimize(d1, initial_x, LBFGS())
    Optim_gf.optimize(d2, initial_x, LBFGS())

    Optim_gf.optimize(f, initial_x, NelderMead())
    ne_res = Optim_gf.optimize(f, initial_x, NelderMead(), Optim_gf.Options(store_trace = true))
    @test_throws ErrorException Optim_gf.simplex_trace(ne_res)
    @test_throws ErrorException Optim_gf.simplex_value_trace(ne_res)
    @test_throws ErrorException Optim_gf.centroid_trace(ne_res)
    ne_res2 = Optim_gf.optimize(
        f,
        initial_x,
        NelderMead(),
        Optim_gf.Options(store_trace = true, trace_simplex = true),
    )
    Optim_gf.simplex_trace(ne_res2)
    Optim_gf.simplex_value_trace(ne_res2)
    @test_throws ErrorException Optim_gf.centroid_trace(ne_res2)
    ne_res3 = Optim_gf.optimize(
        f,
        initial_x,
        NelderMead(),
        Optim_gf.Options(store_trace = true, extended_trace = true, trace_simplex = true),
    )
    Optim_gf.simplex_trace(ne_res3)
    Optim_gf.simplex_value_trace(ne_res3)
    Optim_gf.centroid_trace(ne_res3)

    Optim_gf.optimize(d3, initial_x, Newton())

    Optim_gf.optimize(f, initial_x, SimulatedAnnealing())

    optimize(f, initial_x, BFGS())
    optimize(f, g!, initial_x, BFGS())
    optimize(f, g!, h!, initial_x, BFGS())

    optimize(f, initial_x, GradientDescent())
    optimize(f, g!, initial_x, GradientDescent())
    optimize(f, g!, h!, initial_x, GradientDescent())

    optimize(f, initial_x, LBFGS())
    optimize(f, g!, initial_x, LBFGS())
    optimize(f, g!, h!, initial_x, LBFGS())

    optimize(f, initial_x, NelderMead())
    optimize(f, g!, initial_x, NelderMead())
    optimize(f, g!, h!, initial_x, NelderMead())

    optimize(f, g!, h!, initial_x, Newton())

    optimize(f, initial_x, SimulatedAnnealing())
    optimize(f, g!, initial_x, SimulatedAnnealing())
    optimize(f, g!, h!, initial_x, SimulatedAnnealing())

    options = Optim_gf.Options(
        g_abstol = 1e-12,
        iterations = 10,
        store_trace = true,
        show_trace = false,
    )
    res = optimize(f, g!, h!, initial_x, BFGS(), options)

    options_g = Optim_gf.Options(
        g_abstol = 1e-12,
        iterations = 10,
        store_trace = true,
        show_trace = false,
    )
    options_f = Optim_gf.Options(
        g_abstol = 1e-12,
        iterations = 10,
        store_trace = true,
        show_trace = false,
    )

    res = optimize(f, g!, h!, initial_x, GradientDescent(), options_g)

    res = optimize(f, g!, h!, initial_x, LBFGS(), options_g)

    res = optimize(f, g!, h!, initial_x, NelderMead(), options_f)

    res = optimize(f, g!, h!, initial_x, Newton(), options_g)
    options_sa = Optim_gf.Options(iterations = 10, store_trace = true, show_trace = false)
    res = optimize(f, g!, h!, initial_x, SimulatedAnnealing(), options_sa)

    res = optimize(f, g!, h!, initial_x, BFGS(), options_g)
    options_ext = Optim_gf.Options(
        g_abstol = 1e-12,
        iterations = 10,
        store_trace = true,
        show_trace = false,
        extended_trace = true,
    )
    res_ext = optimize(f, g!, h!, initial_x, BFGS(), options_ext)

    test_summary(res, "BFGS")
    @test Optim_gf.minimum(res) ≈ 1.4015611545580768
    @test Optim_gf.minimizer(res) ≈ [-0.18088789037641684, 0.02431510621380093] rtol = 0.001
    @test Optim_gf.iterations(res) == 10
    @test Optim_gf.f_calls(res) in (18, 19,)
    @test Optim_gf.g_calls(res) in (18, 19,)
    @test !Optim_gf.converged(res)
    @test !Optim_gf.x_converged(res)
    @test !Optim_gf.f_converged(res)
    @test !Optim_gf.g_converged(res)
    @test Optim_gf.g_abstol(res) == 1e-12
    @test Optim_gf.iteration_limit_reached(res)
    @test Optim_gf.initial_x(res) == [-1.2, 1.0]
    @test haskey(Optim_gf.trace(res_ext)[1].metadata, "x")
    @test Optim_gf.termination_code(res_ext) == Optim_gf.TerminationCode.Iterations
    # just testing if it runs
    Optim_gf.trace(res)
    Optim_gf.f_trace(res)
    Optim_gf.g_norm_trace(res)
    @test_throws ErrorException Optim_gf.x_trace(res)
    @test_throws ErrorException Optim_gf.x_lower_trace(res)
    @test_throws ErrorException Optim_gf.x_upper_trace(res)
    @test_throws ErrorException Optim_gf.lower_bound(res)
    @test_throws ErrorException Optim_gf.upper_bound(res)
    @test_throws ErrorException Optim_gf.rel_tol(res)
    @test_throws ErrorException Optim_gf.abs_tol(res)
    options_extended = Optim_gf.Options(store_trace = true, extended_trace = true)
    res_extended = Optim_gf.optimize(f, g!, initial_x, BFGS(), options_extended)
    @test haskey(Optim_gf.trace(res_extended)[1].metadata, "~inv(H)")
    @test haskey(Optim_gf.trace(res_extended)[1].metadata, "g(x)")
    @test haskey(Optim_gf.trace(res_extended)[1].metadata, "x")
    options_extended_nm = Optim_gf.Options(store_trace = true, extended_trace = true)
    res_extended_nm = Optim_gf.optimize(f, g!, initial_x, NelderMead(), options_extended_nm)
    @test haskey(Optim_gf.trace(res_extended_nm)[1].metadata, "centroid")
    @test haskey(Optim_gf.trace(res_extended_nm)[1].metadata, "step_type")



    resgterm = optimize(f, g!, initial_x, BFGS(), Optim_gf.Options(g_abstol = 1e12))
    Optim_gf.termination_code(resgterm) == Optim_gf.TerminationCode.GradientNorm

    resx0term = optimize(
        f,
        g!,
        initial_x,
        BFGS(),
        Optim_gf.Options(
            x_reltol = -1,
            g_abstol = -1,
            f_abstol = -1,
            f_reltol = -1,
            iterations = 10^10,
            allow_f_increases = true,
        ),
    )
    Optim_gf.termination_code(resx0term) == Optim_gf.TerminationCode.NoXChange


    resx0term1 = optimize(
        f,
        g!,
        initial_x,
        BFGS(),
        Optim_gf.Options(
            x_abstol = -1,
            g_abstol = -1,
            f_abstol = -1,
            f_reltol = -1,
            iterations = 10^10,
            allow_f_increases = true,
        ),
    )
    Optim_gf.termination_code(resx0term1) == Optim_gf.TerminationCode.NoXChange


    resxsmallterm1 = optimize(
        f,
        g!,
        initial_x,
        BFGS(),
        Optim_gf.Options(
            x_abstol = 1e-3,
            x_reltol = -1,
            g_abstol = -1,
            f_abstol = -1,
            f_reltol = -1,
            iterations = 10^10,
            allow_f_increases = true,
        ),
    )
    Optim_gf.termination_code(resxsmallterm1) == Optim_gf.TerminationCode.SmallXChange
    resxsmallterm2 = optimize(
        f,
        g!,
        initial_x,
        BFGS(),
        Optim_gf.Options(
            x_abstol = -1,
            x_reltol = 1e-3,
            g_abstol = -1,
            f_abstol = -1,
            f_reltol = -1,
            iterations = 10^10,
            allow_f_increases = true,
        ),
    )
    Optim_gf.termination_code(resxsmallterm2) == Optim_gf.TerminationCode.SmallXChange

end

# Test univariate API
@testset "Univariate API" begin
    f(x) = 2x^2 + 3x + 1
    res = optimize(f, -2.0, 1.0, GoldenSection())
    test_summary(res, "Golden Section Search")
    @test Optim_gf.minimum(res) ≈ -0.125
    @test Optim_gf.minimizer(res) ≈ -0.749999994377939
    @test Optim_gf.iterations(res) == 38
    @test !Optim_gf.iteration_limit_reached(res)
    @test_throws ErrorException Optim_gf.trace(res)
    @test_throws ErrorException Optim_gf.x_trace(res)
    @test_throws ErrorException Optim_gf.x_lower_trace(res)
    @test_throws ErrorException Optim_gf.x_upper_trace(res)
    @test_throws ErrorException Optim_gf.f_trace(res)
    @test Optim_gf.lower_bound(res) == -2.0
    @test Optim_gf.upper_bound(res) == 1.0
    @test Optim_gf.rel_tol(res) ≈ 1.4901161193847656e-8
    @test Optim_gf.abs_tol(res) ≈ 2.220446049250313e-16
    @test_throws ErrorException Optim_gf.initial_x(res)
    @test_throws ErrorException Optim_gf.g_norm_trace(res)
    @test_throws ErrorException Optim_gf.g_calls(res)
    @test_throws ErrorException Optim_gf.x_converged(res)
    @test_throws ErrorException Optim_gf.f_converged(res)
    @test_throws ErrorException Optim_gf.g_converged(res)
    @test_throws ErrorException Optim_gf.x_tol(res)
    @test_throws ErrorException Optim_gf.f_tol(res)
    @test_throws ErrorException Optim_gf.g_abstol(res)
    res = optimize(f, -2.0, 1.0, GoldenSection(), store_trace = true, extended_trace = true)

    # Right now, these just "test" if they run
    Optim_gf.x_trace(res)
    Optim_gf.x_lower_trace(res)
    Optim_gf.x_upper_trace(res)
end

@testset "#948" begin
    res1 = optimize(OnceDifferentiable(t -> t[1]^2, (t, g) -> fill!(g, NaN), [0.0]), [0.0])
    res2 = optimize(OnceDifferentiable(t -> t[1]^2, (t, g) -> fill!(g, Inf), [0.0]), [0.0])
    res3 = optimize(OnceDifferentiable(t -> Inf, [0.0]), [0.0])
    @test !Optim_gf.converged(res1)
    @test !Optim_gf.converged(res2)
    @test !Optim_gf.converged(res3)
end
