using BlockArrays:
  AbstractBlockArray, Block, BlockedOneTo, blockedrange, blocklengths, blocksize
using BlockSparseArrays:
  BlockSparseArray, BlockSparseMatrix, BlockSparseVector, blockstoredlength
using GradedArrays:
  GradedArray,
  GradedMatrix,
  GradedVector,
  GradedOneTo,
  GradedUnitRange,
  U1,
  dag,
  dual,
  gradedrange,
  isdual
using SparseArraysBase: storedlength
using LinearAlgebra: adjoint
using Random: randn!
using Test: @test, @testset

function randn_blockdiagonal(elt::Type, axes::Tuple)
  a = BlockSparseArray{elt}(undef, axes)
  blockdiaglength = minimum(blocksize(a))
  for i in 1:blockdiaglength
    b = Block(ntuple(Returns(i), ndims(a)))
    a[b] = randn!(a[b])
  end
  return a
end

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "GradedArray (eltype=$elt)" for elt in elts
  @testset "definitions" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    a = BlockSparseArray{elt}(undef)
    @test !(a isa GradedArray)  # no type piracy
    v = BlockSparseArray{elt}(undef, r)
    @test v isa GradedArray
    @test v isa GradedVector
    m = BlockSparseArray{elt}(undef, r, r)
    @test m isa GradedArray
    @test m isa GradedMatrix
    a = BlockSparseArray{elt}(undef, r, r, r)
    @test a isa GradedArray
  end
  @testset "map" begin
    d1 = gradedrange([U1(0) => 2, U1(1) => 2])
    d2 = gradedrange([U1(0) => 2, U1(1) => 2])
    a = randn_blockdiagonal(elt, (d1, d2, d1, d2))
    @test axes(a, 1) isa GradedOneTo
    @test axes(view(a, 1:4, 1:4, 1:4, 1:4), 1) isa GradedOneTo

    d1 = gradedrange([U1(0) => 2, U1(1) => 2])
    d2 = gradedrange([U1(0) => 2, U1(1) => 2])
    a = randn_blockdiagonal(elt, (d1, d2, d1, d2))
    for b in (a + a, 2 * a)
      @test size(b) == (4, 4, 4, 4)
      @test blocksize(b) == (2, 2, 2, 2)
      @test blocklengths.(axes(b)) == ([2, 2], [2, 2], [2, 2], [2, 2])
      @test storedlength(b) == 32
      @test blockstoredlength(b) == 2
      for i in 1:ndims(a)
        @test axes(b, i) isa GradedOneTo
      end
      @test label(axes(b, 1)[Block(1)]) == U1(0)
      @test label(axes(b, 1)[Block(2)]) == U1(1)
      @test Array(b) isa Array{elt}
      @test Array(b) == b
      @test 2 * Array(a) == b
    end

    r = gradedrange([U1(0) => 2, U1(1) => 2])
    a = zeros(r, r, r, r)
    @test a isa BlockSparseArray{Float64}
    @test a isa GradedArray
    @test eltype(a) === Float64
    @test size(a) == (4, 4, 4, 4)
    @test iszero(a)
    @test iszero(blockstoredlength(a))

    r = gradedrange([U1(0) => 2, U1(1) => 2])
    a = zeros(elt, r, r, r, r)
    @test a isa BlockSparseArray{elt}
    @test eltype(a) === elt
    @test size(a) == (4, 4, 4, 4)
    @test iszero(a)
    @test iszero(blockstoredlength(a))

    r = gradedrange([U1(0) => 2, U1(1) => 2])
    a = randn_blockdiagonal(elt, (r, r, r, r))
    b = similar(a, ComplexF64)
    @test b isa BlockSparseArray{ComplexF64}
    @test eltype(b) === ComplexF64

    a = BlockSparseVector{Float64}(undef, gradedrange([U1(0) => 1, U1(1) => 1]))
    b = similar(a, Float32)
    @test b isa BlockSparseVector{Float32}
    @test eltype(b) == Float32

    # Test mixing graded axes and dense axes
    # in addition/broadcasting.
    d1 = gradedrange([U1(0) => 2, U1(1) => 2])
    d2 = gradedrange([U1(0) => 2, U1(1) => 2])
    a = randn_blockdiagonal(elt, (d1, d2, d1, d2))
    for b in (a + Array(a), Array(a) + a)
      @test size(b) == (4, 4, 4, 4)
      @test blocksize(b) == (2, 2, 2, 2)
      @test blocklengths.(axes(b)) == ([2, 2], [2, 2], [2, 2], [2, 2])
      @test storedlength(b) == 256
      @test blockstoredlength(b) == 16
      for i in 1:ndims(a)
        @test axes(b, i) isa BlockedOneTo{Int}
      end
      @test Array(a) isa Array{elt}
      @test Array(a) == a
      @test 2 * Array(a) == b
    end

    d1 = gradedrange([U1(0) => 2, U1(1) => 2])
    d2 = gradedrange([U1(0) => 2, U1(1) => 2])
    a = randn_blockdiagonal(elt, (d1, d2, d1, d2))
    b = a[2:3, 2:3, 2:3, 2:3]
    @test size(b) == (2, 2, 2, 2)
    @test blocksize(b) == (2, 2, 2, 2)
    @test storedlength(b) == 2
    @test blockstoredlength(b) == 2
    for i in 1:ndims(a)
      @test axes(b, i) isa GradedOneTo
    end
    @test label(axes(b, 1)[Block(1)]) == U1(0)
    @test label(axes(b, 1)[Block(2)]) == U1(1)
    @test Array(a) isa Array{elt}
    @test Array(a) == a
  end

  @testset "dual axes" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    for ax in ((r, r), (dual(r), r), (r, dual(r)), (dual(r), dual(r)))
      a = BlockSparseArray{elt}(undef, ax...)
      @views for b in [Block(1, 1), Block(2, 2)]
        a[b] = randn(elt, size(a[b]))
      end
      for dim in 1:ndims(a)
        @test typeof(ax[dim]) === typeof(axes(a, dim))
        @test isdual(ax[dim]) == isdual(axes(a, dim))
      end
      @test @view(a[Block(1, 1)])[1, 1] == a[1, 1]
      @test @view(a[Block(1, 1)])[2, 1] == a[2, 1]
      @test @view(a[Block(1, 1)])[1, 2] == a[1, 2]
      @test @view(a[Block(1, 1)])[2, 2] == a[2, 2]
      @test @view(a[Block(2, 2)])[1, 1] == a[3, 3]
      @test @view(a[Block(2, 2)])[2, 1] == a[4, 3]
      @test @view(a[Block(2, 2)])[1, 2] == a[3, 4]
      @test @view(a[Block(2, 2)])[2, 2] == a[4, 4]
      @test @view(a[Block(1, 1)])[1:2, 1:2] == a[1:2, 1:2]
      @test @view(a[Block(2, 2)])[1:2, 1:2] == a[3:4, 3:4]
      a_dense = Array(a)
      @test eachindex(a) == CartesianIndices(size(a))
      for I in eachindex(a)
        @test a[I] == a_dense[I]
      end
      @test axes(a') == dual.(reverse(axes(a)))

      @test isdual(axes(a', 1)) ≠ isdual(axes(a, 2))
      @test isdual(axes(a', 2)) ≠ isdual(axes(a, 1))
      @test isnothing(show(devnull, MIME("text/plain"), a))

      # Check preserving dual in tensor algebra.
      for b in (a + a, 2 * a, 3 * a - a)
        @test Array(b) ≈ 2 * Array(a)
        for dim in 1:ndims(a)
          @test isdual(axes(b, dim)) == isdual(axes(a, dim))
        end
      end

      @test isnothing(show(devnull, MIME("text/plain"), @view(a[Block(1, 1)])))
      @test @view(a[Block(1, 1)]) == a[Block(1, 1)]
    end

    @testset "GradedOneTo" begin
      r = gradedrange([U1(0) => 2, U1(1) => 2])
      a = BlockSparseArray{elt}(undef, r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test blockstoredlength(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedOneTo
        @test axes(a[:, :], i) isa GradedOneTo
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test a[:, I] isa AbstractBlockArray
      @test size(a[I, I]) == (1, 1)
      @test !isdual(axes(a[I, I], 1))
    end

    @testset "GradedUnitRange" begin
      r = gradedrange([U1(0) => 2, U1(1) => 2])[1:3]
      a = BlockSparseArray{elt}(undef, r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test blockstoredlength(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedUnitRange
        @test axes(a[:, :], i) isa GradedUnitRange
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test axes(a[I, :], 1) isa GradedOneTo
      @test axes(a[I, :], 2) isa GradedUnitRange

      @test a[:, I] isa AbstractBlockArray
      @test axes(a[:, I], 2) isa GradedOneTo
      @test axes(a[:, I], 1) isa GradedUnitRange
      @test size(a[I, I]) == (1, 1)
      @test !isdual(axes(a[I, I], 1))
    end

    # Test case when all axes are dual.
    @testset "dual GradedOneTo" begin
      r = gradedrange([U1(-1) => 2, U1(1) => 2])
      a = BlockSparseArray{elt}(undef, dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test blockstoredlength(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedUnitRangeDual
        @test axes(a[:, :], i) isa GradedUnitRangeDual
      end
      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test a[:, I] isa AbstractBlockArray
      @test size(a[I, I]) == (1, 1)
      @test isdual(axes(a[I, :], 2))
      @test isdual(axes(a[:, I], 1))
      @test isdual(axes(a[I, :], 1))
      @test isdual(axes(a[:, I], 2))
      @test isdual(axes(a[I, I], 1))
      @test isdual(axes(a[I, I], 2))
    end

    @testset "dual GradedUnitRange" begin
      r = gradedrange([U1(0) => 2, U1(1) => 2])[1:3]
      a = BlockSparseArray{elt}(undef, dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test blockstoredlength(b) == 2
      @test Array(b) == 2 * Array(a)
      for i in 1:2
        @test axes(b, i) isa GradedUnitRangeDual
        @test axes(a[:, :], i) isa GradedUnitRangeDual
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa AbstractBlockArray
      @test a[:, I] isa AbstractBlockArray
      @test size(a[I, I]) == (1, 1)
      @test isdual(axes(a[I, :], 2))
      @test isdual(axes(a[:, I], 1))
      @test isdual(axes(a[I, :], 1))
      @test isdual(axes(a[:, I], 2))
      @test isdual(axes(a[I, I], 1))
      @test isdual(axes(a[I, I], 2))
    end

    @testset "dual BlockedUnitRange" begin    # self dual
      r = blockedrange([2, 2])
      a = BlockSparseArray{elt}(undef, dual(r), dual(r))
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a
      @test blockstoredlength(b) == 2
      @test Array(b) == 2 * Array(a)
      @test a[:, :] isa BlockSparseArray
      for i in 1:2
        @test axes(b, i) isa BlockedOneTo
        @test axes(a[:, :], i) isa BlockedOneTo
      end

      I = [Block(1)[1:1]]
      @test a[I, :] isa BlockSparseArray
      @test a[:, I] isa BlockSparseArray
      @test size(a[I, I]) == (1, 1)
      @test !isdual(axes(a[I, I], 1))
    end

    # Test case when all axes are dual from taking the adjoint.
    for r in (
      gradedrange([U1(0) => 2, U1(1) => 2]),
      gradedrange([U1(0) => 2, U1(1) => 2])[begin:end],
    )
      a = BlockSparseArray{elt}(undef, r, r)
      @views for i in [Block(1, 1), Block(2, 2)]
        a[i] = randn(elt, size(a[i]))
      end
      b = 2 * a'
      @test blockstoredlength(b) == 2
      @test Array(b) == 2 * Array(a)'
      for ax in axes(b)
        @test ax isa typeof(dual(r))
      end

      @test !isdual(axes(a, 1))
      @test !isdual(axes(a, 2))
      @test isdual(axes(a', 1))
      @test isdual(axes(a', 2))
      @test isdual(axes(b, 1))
      @test isdual(axes(b, 2))
      @test isdual(axes(copy(a'), 1))
      @test isdual(axes(copy(a'), 2))

      I = [Block(1)[1:1]]
      @test size(b[I, :]) == (1, 4)
      @test size(b[:, I]) == (4, 1)
      @test size(b[I, I]) == (1, 1)
    end
  end
  @testset "Matrix multiplication" begin
    r = gradedrange([U1(0) => 2, U1(1) => 3])
    a1 = BlockSparseArray{elt}(undef, dual(r), r)
    a1[Block(1, 2)] = randn(elt, size(@view(a1[Block(1, 2)])))
    a1[Block(2, 1)] = randn(elt, size(@view(a1[Block(2, 1)])))
    a2 = BlockSparseArray{elt}(undef, dual(r), r)
    a2[Block(1, 2)] = randn(elt, size(@view(a2[Block(1, 2)])))
    a2[Block(2, 1)] = randn(elt, size(@view(a2[Block(2, 1)])))
    @test Array(a1 * a2) ≈ Array(a1) * Array(a2)
    @test Array(a1' * a2') ≈ Array(a1') * Array(a2')

    a2 = BlockSparseArray{elt}(undef, r, dual(r))
    a2[Block(1, 2)] = randn(elt, size(@view(a2[Block(1, 2)])))
    a2[Block(2, 1)] = randn(elt, size(@view(a2[Block(2, 1)])))
    @test Array(a1' * a2) ≈ Array(a1') * Array(a2)
    @test Array(a1 * a2') ≈ Array(a1) * Array(a2')
  end
  @testset "Construct from dense" begin
    r = gradedrange([U1(0) => 2, U1(1) => 3])
    a1 = randn(elt, 2, 2)
    a2 = randn(elt, 3, 3)
    a = cat(a1, a2; dims=(1, 2))
    b = a[r, dual(r)]
    @test eltype(b) === elt
    @test b isa BlockSparseMatrix{elt}
    @test blockstoredlength(b) == 2
    @test b[Block(1, 1)] == a1
    @test iszero(b[Block(2, 1)])
    @test iszero(b[Block(1, 2)])
    @test b[Block(2, 2)] == a2
    @test all(GradedUnitRanges.space_isequal.(axes(b), (r, dual(r))))

    # Regression test for Vector, which caused
    # an ambiguity error with Base.
    r = gradedrange([U1(0) => 2, U1(1) => 3])
    a1 = randn(elt, 2)
    a2 = zeros(elt, 3)
    a = vcat(a1, a2)
    b = a[r]
    @test eltype(b) === elt
    @test b isa BlockSparseVector{elt}
    @test blockstoredlength(b) == 1
    @test b[Block(1)] == a1
    @test iszero(b[Block(2)])
    @test all(GradedUnitRanges.space_isequal.(axes(b), (r,)))

    # Regression test for BitArray
    r = gradedrange([U1(0) => 2, U1(1) => 3])
    a1 = trues(2, 2)
    a2 = trues(3, 3)
    a = cat(a1, a2; dims=(1, 2))
    b = a[r, dual(r)]
    @test eltype(b) === Bool
    @test b isa BlockSparseMatrix{Bool}
    @test blockstoredlength(b) == 2
    @test b[Block(1, 1)] == a1
    @test iszero(b[Block(2, 1)])
    @test iszero(b[Block(1, 2)])
    @test b[Block(2, 2)] == a2
    @test all(GradedUnitRanges.space_isequal.(axes(b), (r, dual(r))))
  end
end

@testset "dag" begin
  elt = ComplexF64
  r = gradedrange([U1(0) => 2, U1(1) => 3])
  a = BlockSparseArray{elt}(undef, r, dual(r))
  a[Block(1, 1)] = randn(elt, 2, 2)
  a[Block(2, 2)] = randn(elt, 3, 3)
  @test isdual.(axes(a)) == (false, true)
  ad = dag(a)
  @test Array(ad) == conj(Array(a))
  @test isdual.(axes(ad)) == (true, false)
end
