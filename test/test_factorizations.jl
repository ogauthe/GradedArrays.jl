using BlockArrays: Block
using GradedArrays: U1, gradedrange, dual
using TensorAlgebra: svd
using Test: @test, @testset

@testset "SVD" begin
  g = gradedrange([U1(1) => 2])
  m = zeros(g, dual(g))
  m[Block(1, 1)] = randn(2, 2)
  u, s, v = svd(m, (1, 2), (1,), (2,))

  @test_broken u isa GradedArray
  @test_broken s isa GradedArray
  @test_broken v isa GradedArray

  @test_broken !(axes(s) isa Tuple{BlockedOneTo,BlockedOneTo})
end
