
# ==========================================================================================
# explicitly defining identity_dimension
# allows to define length, but less generic
struct KroneckerRange{T<:Integer,Range<:AbstractUnitRange{T}} <: AbstractUnitRange{T}
  identity_dimension::T
  inner_range::Range
  function KroneckerRange{T,Range}(identity_dimension, r) where {T,Range}
    return new{T,Range}(identity_dimension, r)
  end
end

# accessors
inner_range(kr::KroneckerRange) = kr.inner_range
identity_dimension(kr::KroneckerRange) = kr.identity_dimension

# constructors
kroneckerrange(a, r) = KroneckerRange{eltype(r),typeof(r)}(a, r)

# Base interface
function Base.length(kr::KroneckerRange)
  return identity_dimension(kr) * length(inner_range)
end

function Base.getindex(sr::SectorUnitRange, kr::KroneckerRange)
  @assert length(sector(sr)) == identity_dimension(kr)
  return sr[(:, inner_range(kr))]
end

function Base.getindex(g::GradedUnitRange, bkr::BlockIndexRange{1,Tuple{KroneckerRange}})
  return g[bkr.block][bkr.indices]
end

function Base.show(io::IO, kr::KroneckerRange)
  return println(io, nameof(typeof(kr)), " ", identity_dimension(kr), "×", inner_range(kr))
end

#=
struct CartesianProduct{A,B}
    a::A
    b::B
end

arguments(cp::CartesianProduct) = (cp.a, cp.b)

function Base.getindex(cp1::CartesianProduct,cp2::CartesianProduct)
    cpa1, cpb1 = arguments(cp1)
    cpa2, cpb2 = arguments(cp2)
    return CartesianProduct(cpa1[cpa2], cpb1[cpb2])
end

kr = CartesianProduct(SU2(1), 2:3)
kr = CartesianProduct(:, 2:3)

struct BlockCartesianRange{N,I}
    block::NTuple{N,Int}
    indices::I
end
bkr = Block(1,1)[kr, kr]
=#
