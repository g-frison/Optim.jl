@testset "Complex numbers" begin
    Random.seed!(0)

    # Test case: minimize quadratic plus quartic
    # μ is the strength of the quartic. μ = 0 is just a quadratic problem
    n = 4
    A = randn(n, n) + im * randn(n, n)
    A = A'A + I
    b = randn(n) + im * randn(n)
    μ = 1.0

    fcomplex(x) = real(dot(x, A * x) / 2 - dot(b, x)) + μ * sum(abs.(x) .^ 4)
    gcomplex(x) = A * x - b + 4μ * (abs.(x) .^ 2) .* x
    gcomplex!(stor, x) = copyto!(stor, gcomplex(x))


    x0 = randn(n) + im * randn(n)

    xref = Optim_gf.minimizer(Optim_gf.optimize(fcomplex, gcomplex!, x0, Optim_gf.LBFGS()))

    @testset "Finite difference setup" begin
        oda1 = OnceDifferentiable(fcomplex, x0)
        fx, gx = NLSolversBase.value_gradient!(oda1, x0)
        @test fx == fcomplex(x0)
        @test gcomplex(x0) ≈ NLSolversBase.gradient(oda1)
    end

    @testset "Zeroth and second order methods" begin
        for method in (Optim_gf.NelderMead, Optim_gf.ParticleSwarm, Optim_gf.Newton)
            @test_throws Any Optim_gf.optimize(fcomplex, gcomplex!, x0, method())
        end
        #not supposed to converge, but it should at least go through without errors
        res = Optim_gf.optimize(fcomplex, gcomplex!, x0, Optim_gf.SimulatedAnnealing())
    end


    @testset "First order methods" begin
        options = Optim_gf.Options(allow_f_increases = true)
        # TODO: AcceleratedGradientDescent fail to converge?
        for method in (
            Optim_gf.GradientDescent(),
            Optim_gf.ConjugateGradient(),
            Optim_gf.LBFGS(),
            Optim_gf.BFGS(),
            Optim_gf.NGMRES(),
            Optim_gf.OACCEL(),
            Optim_gf.MomentumGradientDescent(mu = 0.1),
        )

            debug_printing && printstyled("Solver: $(summary(method))\n", color = :green)
            res = Optim_gf.optimize(fcomplex, gcomplex!, x0, method, options)
            debug_printing && printstyled("Iter\tf-calls\tg-calls\n", color = :green)
            debug_printing && printstyled(
                "$(Optim_gf.iterations(res))\t$(Optim_gf.f_calls(res))\t$(Optim_gf.g_calls(res))\n",
                color = :red,
            )
            if !Optim_gf.converged(res)
                @warn("$(summary(method)) failed.")
                display(res)
                println("########################")
            end
            test_summary(res) # Just check that no errors arise when doing display(res)
            @test typeof(fcomplex(x0)) == typeof(Optim_gf.minimum(res))
            @test eltype(x0) == eltype(Optim_gf.minimizer(res))
            @test Optim_gf.converged(res)
            @test Optim_gf.minimizer(res) ≈ xref rtol = 1e-4

            res = Optim_gf.optimize(fcomplex, x0, method, options)
            @test Optim_gf.converged(res)
            @test Optim_gf.minimizer(res) ≈ xref rtol = 1e-4

            # # To compare with the equivalent real solvers
            # to_cplx(x) = x[1:n] + im*x[n+1:2n]
            # from_cplx(x) = [real(x);imag(x)]
            # freal(x) = fcomplex(to_cplx(x))
            # greal!(stor,x) = copyto!(stor, from_cplx(gcomplex(to_cplx(x))))
            # opt = Optim_gf.Options(allow_f_increases=true,show_trace=true)
            # println("$(summary(method)) cplx")
            # res_cplx = Optim_gf.optimize(fcomplex,gcomplex!,x0,method,opt)
            # println("$(summary(method)) real")
            # res_real = Optim_gf.optimize(freal,greal!,from_cplx(x0),method,opt)
        end
    end
end
