using ITensors: contract

mutable struct ReducedLinearProblem <: AbstractProjMPO
    reduced_operator::ProjMPO
    reduced_constant_terms::Vector{ReducedConstantTerm}
end

# Linear problem updater interface.
operator(reduced_problem::ReducedLinearProblem) = reduced_problem.reduced_operator
function constant_term(reduced_problem::ReducedLinearProblem)
    constant_terms = map(reduced_problem.reduced_constant_terms) do reduced_constant_term
        return contract(reduced_constant_term)
    end
    return dag(only(constant_terms))
end

function ReducedLinearProblem(operator::MPO, constant_term::MPS)
    return ReducedLinearProblem(ProjMPO(operator), [ReducedConstantTerm(constant_term)])
end

function ReducedLinearProblem(operator::MPO, constant_terms::Vector{MPS})
    return ReducedLinearProblem(ProjMPO(operator), ReducedConstantTerm.(constant_terms))
end

function Base.copy(reduced_problem::ReducedLinearProblem)
    return ReducedLinearProblem(
        copy(reduced_problem.reduced_operator), copy(reduced_problem.reduced_constant_terms)
    )
end

function ITensorMPS.nsite(reduced_problem::ReducedLinearProblem)
    return nsite(reduced_problem.reduced_operator)
end

function ITensorMPS.set_nsite!(reduced_problem::ReducedLinearProblem, nsite)
    set_nsite!(reduced_problem.reduced_operator, nsite)
    for m in reduced_problem.reduced_constant_terms
        set_nsite!(m, nsite)
    end
    return reduced_problem
end

function ITensorMPS.makeL!(reduced_problem::ReducedLinearProblem, state::MPS, position::Int)
    makeL!(reduced_problem.reduced_operator, state, position)
    for reduced_constant_term in reduced_problem.reduced_constant_terms
        makeL!(reduced_constant_term, state, position)
    end
    return reduced_problem
end

function ITensorMPS.makeR!(reduced_problem::ReducedLinearProblem, state::MPS, position::Int)
    makeR!(reduced_problem.reduced_operator, state, position)
    for reduced_constant_term in reduced_problem.reduced_constant_terms
        makeR!(reduced_constant_term, state, position)
    end
    return reduced_problem
end

function ITensors.contract(reduced_problem::ReducedLinearProblem, v::ITensor)
    return contract(reduced_problem.reduced_operator, v)
end

mutable struct ReducedPrecondLinearProblem <: AbstractProjMPO
    linear_problem::ReducedLinearProblem
    preconditioner::ProjMPO
    residual::MPS
end

# Access the residual MPS (b - Ax) used for building the preconditioner environments.
residual(reduced_problem::ReducedPrecondLinearProblem) = reduced_problem.residual

# Update the residual MPS (b - Ax) used for building the preconditioner environments.
function set_residual!(reduced_problem::ReducedPrecondLinearProblem, residual::MPS)
    reduced_problem.residual = residual
    return reduced_problem
end

function ReducedPrecondLinearProblem(operator::MPO, constant_term::MPS, preconditioner::MPO, residual::MPS=copy(constant_term))
    linear_problem = ReducedLinearProblem(ProjMPO(operator), [ReducedConstantTerm(constant_term)])
    preconditioner = ProjMPO(preconditioner)
    return ReducedPrecondLinearProblem(linear_problem, preconditioner, residual)
end

function ReducedPrecondLinearProblem(operator::MPO, constant_terms::Vector{MPS}, preconditioner::MPO, residual::MPS=copy(first(constant_terms)))
    error("Not implemented")
    # The residual in this case would be Σbᵢ which would have an absurd bond dimension, so...
    linear_problem = ReducedLinearProblem(ProjMPO(operator), ReducedConstantTerm.(constant_terms))
    preconditioner = ProjMPO(preconditioner)
    return ReducedPrecondLinearProblem(linear_problem, preconditioner, residual)
end

function Base.copy(reduced_problem::ReducedPrecondLinearProblem)
    return ReducedPrecondLinearProblem(
        copy(reduced_problem.linear_problem), copy(reduced_problem.preconditioner), residual = copy(reduced_problem.residual)
    )
end

function ITensorMPS.nsite(reduced_problem::ReducedPrecondLinearProblem)
    return nsite(reduced_problem.linear_problem)
end

function ITensorMPS.set_nsite!(reduced_problem::ReducedPrecondLinearProblem, nsite)
    set_nsite!(reduced_problem.linear_problem, nsite)
    set_nsite!(reduced_problem.preconditioner, nsite)
    return reduced_problem
end

function ITensorMPS.makeL!(reduced_problem::ReducedPrecondLinearProblem, state::MPS, position::Int)
    makeL!(reduced_problem.linear_problem, state, position)
    makeL!(reduced_problem.preconditioner, reduced_problem.residual, position)
    return reduced_problem
end

function ITensorMPS.makeR!(reduced_problem::ReducedPrecondLinearProblem, state::MPS, position::Int)
    makeR!(reduced_problem.linear_problem, state, position)
    makeR!(reduced_problem.preconditioner, reduced_problem.residual, position)
    return reduced_problem
end