# ====================================  Definitions  =======================================

struct KroneckerMatrix{A<:AbstractMatrix,B<:AbstractMatrix,T} <: AbstractMatrix{T}
  a::A
  b::B
  function KroneckerMatrix{A,B,T}(a, b) where {A,B,T}
    return new{A,B,T}(a, b)
  end
end

# ====================================  Constructors  ======================================

function KroneckerMatrix(a, b)
  return KroneckerMatrix{typeof(a),typeof(b),Base.promote_op(*, eltype(a), eltype(b))}(a, b)
end

kroneckermatrix(a, b) = KroneckerMatrix(a, b)

# =====================================  Accessors  ========================================

arguments(km::KroneckerMatrix) = (km.a, km.b)

# ==================================  Base interface  ======================================

function Base.axes(km::KroneckerMatrix)
  return (
    CartesianProductUnitRange{1:size(km, 1),CartesianProduct(first.(axes.(arguments(km))))},
    CartesianProductUnitRange{1:size(km, 2),CartesianProduct(last.(axes.(arguments(km))))},
  )
end

function Base.show(io::IO, km::KroneckerMatrix)
  println(io, typeof(km))
  println(io, first(arguments(km)))
  return println(io, last(arguments(km)))
end

function Base.size(km::KroneckerMatrix)
  sizes = size.(arguments(km))
  return (prod(first.(sizes)), prod(last.(sizes)))
end

#Base.getindex(km::KroneckeMatrix) = isabelian(km) ? getindex(::AbelianStyle) : error()
