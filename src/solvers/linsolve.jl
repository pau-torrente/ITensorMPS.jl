using KrylovKit: KrylovKit, linsolve
using LinearAlgebra: I, qr, eigvals, Symmetric

# compute_cond = true converts the reduced & projected operator into a matrix and computes the condition number exactly. It is very costly, but useful for development purposes
function krylov_updater(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    kwargs = filter(p -> first(p) != :compute_cond, kwargs)

    x, info = linsolve(
        operator(problem),
        constant_term(problem),
        init,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )
    if compute_cond
        # op = contract(operator(problem))
        # b = constant_term(problem)

        # rowinds = commoninds(op, b)
        # colinds = uniqueinds(op, b)

        # rowdim = prod(dim.(rowinds))
        # coldim = prod(dim.(colinds))

        # Amat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)

        # eig_solution = eigen(Amat)
        # abs_evalues = sort(abs.(eig_solution.values))
        # op_condition_number = last(abs_evalues) / first(abs_evalues)
        # @show op_condition_number
        # @show first(sort(real.(eig_solution.values)))
        op = contract(operator(problem))
        b_tensor = constant_term(problem)
        rowinds = commoninds(op, b_tensor)
        colinds = uniqueinds(op, b_tensor)
        rowdim = prod(dim.(rowinds))
        coldim = prod(dim.(colinds))
        op_mat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
        op_sym = Symmetric(0.5 * (op_mat + transpose(op_mat)))
        eigs_op = eigvals(op_sym)
        min_op_eig = minimum(eigs_op)

        if min_op_eig < 0
            @error "OPERATOR IS NOT SPD!" min_op_eig pos=problem.reduced_operator.lpos
        end

        @show problem.reduced_operator.lpos, min_op_eig, maximum(eigs_op)
    end
    return x, (; info)
end

function krylov_updater(problem::ReducedPrecondLinearProblem, init; internal_kwargs, coefficients, kwargs...)

    # @show problem.linear_problem.reduced_operator.lpos, problem.linear_problem.reduced_operator.rpos
    # @show problem.preconditioner.lpos, problem.preconditioner.rpos
    
    # # Verify the preconditioner is SPD at THIS bond during the sweep
    # op = contract(operator(problem.linear_problem))
    # prec = contract(problem.preconditioner)
    # b = constant_term(problem.linear_problem)
    
    # rowinds = commoninds(op, b)
    # colinds = uniqueinds(op, b)
    # rowdim = prod(dim.(rowinds))
    # coldim = prod(dim.(colinds))
    
    # op_mat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
    # prec_mat = reshape(array(prec, rowinds..., colinds...), rowdim, coldim)
    
    # product_eigs = eigvals(prec_mat * op_mat)
    # @show minimum(real.(product_eigs)), maximum(real.(product_eigs))
    
    # # Also check: what are the operator eigenvalues?
    # op_eigs = eigvals(op_mat)
    # @show minimum(real.(op_eigs)), maximum(real.(op_eigs))
    
    # # What should M⁻¹ eigenvalues be for perfect preconditioning?
    # @show 1.0/maximum(real.(op_eigs)), 1.0/minimum(real.(op_eigs))

    compute_cond = get(kwargs, :compute_cond, false)
    if compute_cond
        # op = contract(operator(problem.linear_problem))
        # b = constant_term(problem.linear_problem)
        # preconditioner = contract(problem.preconditioner)

        # rowinds = commoninds(op, b)
        # colinds = uniqueinds(op, b)

        # rowdim = prod(dim.(rowinds))
        # coldim = prod(dim.(colinds))

        # Amat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
        # M⁻¹mat = reshape(array(preconditioner, rowinds..., colinds...), rowdim, coldim)

        # eig_solution = eigen(Amat)
        # abs_evalues = sort(abs.(eig_solution.values))
        # op_condition_number = last(abs_evalues) / first(abs_evalues)
        # @show op_condition_number
        # @show first(sort(real.(eig_solution.values)))

        # eig_solution = eigen(M⁻¹mat)
        # abs_evalues = sort(abs.(eig_solution.values))
        # precond_condition_number = last(abs_evalues) / first(abs_evalues)
        # @show precond_condition_number
        # @show first(sort(real.(eig_solution.values)))
        op = contract(operator(problem.linear_problem))
        b_tensor = constant_term(problem.linear_problem)
        rowinds = commoninds(op, b_tensor)
        colinds = uniqueinds(op, b_tensor)
        rowdim = prod(dim.(rowinds))
        coldim = prod(dim.(colinds))
        op_mat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
        op_sym = Symmetric(0.5 * (op_mat + transpose(op_mat)))
        eigs_op = eigvals(op_sym)
        min_op_eig = minimum(eigs_op)

        pc = contract(problem.preconditioner)
        pc_mat = reshape(array(pc, rowinds..., colinds...), rowdim, coldim)
        pc_sym = Symmetric(0.5 * (pc_mat + transpose(pc_mat)))
        eigs_pc = eigvals(pc_sym)
        min_pc_eig = minimum(eigs_pc)

        if min_op_eig < 0
            @error "OPERATOR IS NOT SPD!" min_op_eig pos=problem.linear_problem.reduced_operator.lpos
        end
        
        if min_pc_eig < 0
            @error "Preconditioner IS NOT SPD!" min_pc_eig pos=problem.linear_problem.reduced_operator.lpos
        end

        @show problem.linear_problem.reduced_operator.lpos, min_op_eig, maximum(eigs_op), min_pc_eig, maximum(eigs_pc)
    end

    kwargs = filter(p -> first(p) != :compute_cond, kwargs)
    x, info = linsolve(
        operator(problem.linear_problem),
        constant_term(problem.linear_problem),
        init,
        problem.preconditioner,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )
    sleep(3)
    return x, (; info, residual = info.residual)
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

    return noprime(ITensor(x, colinds...)), (; residual)
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
        const_term::MPS,
        init::MPS,
        preconditioner,
        coefficient1::Number = false,
        coefficient2::Number = true;
        updater = krylov_updater,
        updater_kwargs = (;),
        kwargs...,
    )
    # Provisional bruteforce approach to test performance
    reduced_precond_problem = ReducedPrecondLinearProblem(operator, const_term, preconditioner)
    # preconditioned_operator = apply(preconditioner, operator; maxdim = maxlinkdim(operator))
    # preconditioner_constterm = apply(preconditioner, constant_term; maxdim = maxlinkdim(constant_term))
    # reduced_precond_problem = ReducedLinearProblem(preconditioned_operator, preconditioner_constterm)
    updater_kwargs = (; coefficients = (coefficient1, coefficient2), updater_kwargs...)

    # ...existing code...


    # precond_problem = ReducedPrecondLinearProblem(operator, const_term, preconditioner)

    # set_nsite!(precond_problem, 2)

    # for pos in 1:(length(init)-1)
    #     orthogonalize!(init, pos)
    #     position!(precond_problem, init, pos)
    #     opp = contract(precond_problem.preconditioner)
    #     b = constant_term(precond_problem.linear_problem)
        
    #     rowinds = commoninds(opp, b)
    #     colinds = uniqueinds(opp, b)
    #     rowdim = prod(dim.(rowinds))
    #     coldim = prod(dim.(colinds))
        
    #     opp_mat = reshape(array(opp, rowinds..., colinds...), rowdim, coldim)
    #     opp_eigenvals = eigvals(opp_mat)
    #     min_eig = minimum(opp_eigenvals)
    #     max_eig = maximum(opp_eigenvals)
    #     @show pos, min_eig, max_eig, min_eig/max_eig

    #     # Check symmetry
    #     @show norm(opp_mat - transpose(opp_mat)) / norm(opp_mat)
    #     # Check eigenvalues of the UNSYMMETRIZED matrix
    #     @show minimum(real.(eigvals(opp_mat)))
    #     # Check eigenvalues of the symmetric part
    #     @show minimum(eigvals(Symmetric(0.5 * (opp_mat + transpose(opp_mat)))))

    # end

    return alternating_update(reduced_precond_problem, init; updater, updater_kwargs, kwargs...)
end
