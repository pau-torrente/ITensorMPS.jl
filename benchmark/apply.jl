using Revise
using ITensors
using ITensorMPS
using LinearAlgebra
using BenchmarkTools

sites = siteinds(2, 20)

a = sum(random_mpo(sites) for _ in 1:5)
b = random_mps(sites; linkdims=10)

apply(a, b; alg="densitymatrix")
@benchmark apply(a, b; alg="densitymatrix")