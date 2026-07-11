@testset "inplace keyword" begin
    rosenbrock = MultivariateProblems.UnconstrainedProblems.examples["Rosenbrock"]
    f = MVP.objective(rosenbrock)
    g! = MVP.gradient(rosenbrock)
    h! = MVP.hessian(rosenbrock)
    function g(x)
        G = similar(x)
        g!(G, x)
        G
    end
    function h(x)
        n = length(x)
        H = similar(x, n, n)
        h!(H, x)
        H
    end
    initial_x = rosenbrock.initial_x

    inp_res = optimize(f, g, h, initial_x; inplace = false)
    op_res = optimize(f, g!, h!, initial_x; inplace = true)

    for op in (
        Optim_gf.minimizer,
        Optim_gf.minimum,
        Optim_gf.f_calls,
        Optim_gf.g_calls,
        Optim_gf.jvp_calls,
        Optim_gf.h_calls,
        Optim_gf.hvp_calls,
        Optim_gf.iterations,
        Optim_gf.converged,
    )
        @test all(op(inp_res) .=== op(op_res))
    end
end
