using KrylovKit: KrylovKit, linsolve
using LinearAlgebra: I, qr

function krylov_updater(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    x, info = linsolve(
        operator(problem),
        constant_term(problem),
        init,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )
    return x, (; info)
end

function krylov_updater(problem::ReducedPrecondLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    x, info = linsolve(
        operator(problem.linear_problem),
        constant_term(problem.linear_problem),
	problem.preconditioner,
        init,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )

    return x, (; info)
end

function qr_updater(
    problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...
)
    op = contract(operator(problem))
    b = constant_term(problem)

    rowinds = commoninds(op, b)
    colinds = uniqueinds(op, b)

    rowdim = prod(dim.(rowinds))
    coldim = prod(dim.(colinds))

    bvec = reshape(array(b, rowinds...), rowdim)
    Amat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)

    shifted_Amat = coefficients[1] * I + coefficients[2] * Amat # we are solving (α₁ + α₂A)x = b

    decomp_Amat = qr(shifted_Amat)
    x = decomp_Amat \ bvec

    return noprime(ITensor(x, colinds...)), (;)
end

"""
Compute  a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.

To adjust the balance between accuracy of solution
and speed of the algorithm, it is recommed to first try
adjusting the updater keyword arguments as descibed below.

Keyword arguments:
  - `nsweeps`, `cutoff`, `maxdim`, etc. (like for other MPO/MPS updaters).
  - `updater_kwargs=(;)` - a `NamedTuple` containing keyword arguments that will get forwarded to the local updater,
    in this case `KrylovKit.linsolve` which is a GMRES linear updater. For example:
    ```julia
    linsolve(A, b, x; maxdim=100, cutoff=1e-8, nsweeps=10, updater_kwargs=(; ishermitian=true, tol=1e-6, maxiter=20, krylovdim=30))
    ```
    See `KrylovKit.jl` documentation for more details on available keyword arguments.
"""
# TODO Decide on if we should separate linsolve from KrylovKit if QR turns out to work well. 
# Currently, the updaters are not exported, so this MUST be handled...
function KrylovKit.linsolve(
        operator,
        constant_term::MPS,
        init::MPS,
        coefficient1::Number = false,
        coefficient2::Number = true;
        updater = krylov_updater,
        updater_kwargs = (;),
        kwargs...,
    )
    reduced_problem = ReducedLinearProblem(operator, constant_term)
    updater_kwargs = (; coefficients = (coefficient1, coefficient2), updater_kwargs...)
    return alternating_update(reduced_problem, init; updater, updater_kwargs, kwargs...)
end

function KrylovKit.linsolve(
        operator,
        constant_term::MPS,
        init::MPS,
        preconditioner,
        coefficient1::Number = false,
        coefficient2::Number = true;
        updater = krylov_updater,
        updater_kwargs = (;),
        kwargs...,
    )
    reduced_precond_problem = ReducedPrecondLinearProblem(operator, constant_term, preconditioner)
    updater_kwargs = (; coefficients = (coefficient1, coefficient2), updater_kwargs...)
    return alternating_update(reduced_precond_problem, init; updater, updater_kwargs, kwargs...)
end


