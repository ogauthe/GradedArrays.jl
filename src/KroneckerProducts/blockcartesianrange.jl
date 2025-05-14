using BlockArrays: BlockArrays

# ====================================  Definitions  =======================================
struct BlockCartesianRange{N,I}
  block::Block{N,Int}
  indices::I
end

# =====================================  Accessors  ========================================

_indices(bcr::BlockCartesianRange) = bcr.indices

# ====================================  Constructors  ======================================

blockcartesianrange(b::Block, idx) = BlockCartesianRange(b, idx)

# ==================================  Base interface  ======================================

function Base.getindex(b::Block{N}, indices::Vararg{CartesianProduct,N}) where {N}
  return blockcartesianrange(b, indices)
end

# =============================  BlockArrays interface  ====================================

BlockArrays.block(bcr::BlockCartesianRange) = bcr.block
