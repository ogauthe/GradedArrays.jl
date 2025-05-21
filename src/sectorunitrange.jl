# This files defines SectorUnitRange, a unit range associated with a sector and an arrow

using GradedArrays.KroneckerProducts:
  KroneckerProducts, CartesianProductUnitRange, cartesianproduct, cartesianproductunitrange

# =====================================  Definition  =======================================

# This implementation contains the "full range"
# it does not check that such a range is consistent with the sector quantum_dimension
# when sliced directly, the label is dropped
# when sliced between multiplicities with sr[(:,1:1)], it returns another SectorUnitRange
# TBD impose some compatibility constraints between range and quantum_dimension?

const SectorUnitRange{T,F,Range} = CartesianProductUnitRange{
  T,F,MultRange,Range
} where {T,F<:Flux,MultRange<:AbstractUnitRange{T},Range}
const SectorOneTo{T,F,Range} = CartesianProductUnitRange{
  T,F,MultRange,Base.OneTo{T}
} where {T,F<:Flux,MultRange<:AbstractUnitRange{T}}

# ====================================  Constructors  ======================================

# sectorrange(flux(SU2(1), false), 2:5)
function sectorrange(f::Flux, r::AbstractUnitRange)
  return cartesianproductunitrange(cartesianproduct(f, r), Base.oneto(length(f)))
end

# sectorrange(flux(SU2(1), false), 1)
function sectorrange(f::Flux, m::Integer)
  return sectorrange(f, Base.oneto(m * length(f)))
end

# sectorrange(flux(SU2(1),false) => 1)
function sectorrange(p::Pair)
  return sectorrange(first(p), last(p))
end

# =====================================  Accessors  ========================================

flux(sr::SectorUnitRange) = first(KroneckerProducts.arguments(cartesianproduct(sr)))
sector(sr::SectorUnitRange) = sector(flux(sr))
ungrade(sr::SectorUnitRange) = last(KroneckerProducts.arguments(cartesianproduct(sr)))

# ==================================  Base interface  ======================================

Base.first(sr::SectorUnitRange) = first(ungrade(sr))

Base.iterate(sr::SectorUnitRange) = iterate(ungrade(sr))
Base.iterate(sr::SectorUnitRange, i::Integer) = iterate(ungrade(sr), i)

Base.length(sr::SectorUnitRange) = length(ungrade(sr))

Base.last(sr::SectorUnitRange) = last(ungrade(sr))

# slicing
Base.getindex(sr::SectorUnitRange, i::Integer) = ungrade(sr)[i]

function Base.getindex(sr::SectorUnitRange, r::AbstractUnitRange{T}) where {T<:Integer}
  return sr[SymmetryStyle(sr), r]
end
function Base.getindex(sr::SectorUnitRange, ::NotAbelianStyle, r::AbstractUnitRange)
  return ungrade(sr)[r]
end
function Base.getindex(sr::SectorUnitRange, ::AbelianStyle, r::AbstractUnitRange)
  return sectorrange(sector(sr), ungrade(sr)[r], isdual(sr))
end

# TODO replace (:,x) indexing with kronecker(:, x)
Base.getindex(sr::SectorUnitRange, t::Tuple{Colon,<:Integer}) = sr[(:, last(t):last(t))]
function Base.getindex(sr::SectorUnitRange, t::Tuple{Colon,<:AbstractUnitRange})
  r = last(t)
  new_range = ((first(r) - 1) * length(sector(sr)) + 1):(last(r) * length(sector(sr)))
  return sectorrange(sector(sr), ungrade(sr)[new_range], isdual(sr))
end

function Base.show(io::IO, sr::SectorUnitRange)
  print(io, nameof(typeof(sr)), " ")
  if isdual(sr)
    print(io, "dual(", sector(sr), ")")
  else
    print(io, sector(sr))
  end
  return print(io, " => ", ungrade(sr))
end

# ================================  BlockArrays interface  =================================

# generic

# =============================  GradedUnitRanges interface  ===============================
to_gradedrange(sr::SectorUnitRange) = mortar_axis([sr])

sectors(sr::SectorUnitRange) = [sector(sr)]

function dual(sr::SectorUnitRange)
  return sectorrange(sector(sr), ungrade(sr), !isdual(sr))
end

function flip(sr::SectorUnitRange)
  return sectorrange(dual(sector(sr)), ungrade(sr), !isdual(sr))
end

function map_sectors(f, sr::SectorUnitRange)
  return sectorrange(f(sector(sr)), ungrade(sr), isdual(sr))
end

sector_type(::Type{<:SectorUnitRange{T,Sector}}) where {T,Sector} = Sector

# TBD error for non-integer?
sector_multiplicity(sr::SectorUnitRange) = length(sr) ÷ length(sector(sr))
sector_multiplicities(sr::SectorUnitRange) = [sector_multiplicity(sr)]  # TBD remove?
