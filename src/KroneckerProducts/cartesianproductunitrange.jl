struct CartesianProductUnitRange{T,A,B,Range} <: AbstractUnitRange{T}
  cp::CartesianProduct{A,B}
  full_range::Range
  isdual::Bool

  function CartesianProductUnitRange{T,A,B,Range}(cp, r, b) where {T,A,B,Range}
    return new{T,A,B,Range}(cp, r, b)
  end
end
