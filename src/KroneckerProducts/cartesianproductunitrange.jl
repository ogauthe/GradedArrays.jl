using GradedArrays: GradedArrays, arguments, isdual

# =====================================  Definition  =======================================

struct CartesianProductUnitRange{T,A,B,Range} <: AbstractUnitRange{T}
  cp::CartesianProduct{A,B}
  full_range::Range
  function CartesianProductUnitRange{T,A,B,Range}(cp, r) where {T,A,B,Range}
    return new{T,A,B,Range}(cp, r)
  end
end

# ====================================  Constructors  ======================================

function CartesianProductUnitRange(cp, r)
  A, B = typeof.(arguments(cp))
  return CartesianProductUnitRange{eltype(r),A,B,typeof(r)}(cp, r)
end

cartesianproductunitrange(cp, r) = CartesianProductUnitRange(cp, r)

# =====================================  Accessors  ========================================

cartesianproduct(cpr::CartesianProductUnitRange) = cpr.cp

# ==================================  Base interface  ======================================

Base.axes(cpr::CartesianProductUnitRange) = (Base.oneto(length(cpr.full_range)),)

function Base.show(io::IO, cpr::CartesianProductUnitRange)
  println(io, typeof(cpr))
  println(io, cartesianproduct(cpr))
  return println(io, cpr.full_range, " ")
end

function Base.:(==)(cpr1::CartesianProductUnitRange, cpr2::CartesianProductUnitRange)
  return cartesianproduct(cpr1) == cartesianproduct(cpr2) &&
         cpr1.full_range == cpr2.full_range
end
