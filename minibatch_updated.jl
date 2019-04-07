
using Random
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle

mutable struct Data3d{T}; x1;x2; y; batchsize; length; partial; imax; indices; shuffle; xsize; ysize; x1type;x2type; ytype; end

"""
    minibatch(x, [y], batchsize; shuffle, partial, xtype, ytype, xsize, ysize)

Return an iterator of minibatches [(xi,yi)...] given data tensors x, y and batchsize.  

The last dimension of x and y give the number of instances and should be equal. `y` is
optional, if omitted a sequence of `xi` will be generated rather than `(xi,yi)` tuples.  Use
`repeat(d,n)` for multiple epochs, `Iterators.take(d,n)` for a partial epoch, and
`Iterators.cycle(d)` to cycle through the data forever (this can be used with `converge`).
If you need the iterator to continue from its last position when stopped early (e.g. by a
break in a for loop), use `Iterators.Stateful(d)` (by default the iterator would restart
from the beginning).

Keyword arguments:

- `shuffle=false`: Shuffle the instances every epoch.
- `partial=false`: If true include the last partial minibatch < batchsize.
- `xtype=typeof(x)`: Convert xi in minibatches to this type.
- `ytype=typeof(y)`: Convert yi in minibatches to this type.
- `xsize=size(x)`: Convert xi in minibatches to this shape.
- `ysize=size(y)`: Convert yi in minibatches to this shape.
"""


function minibatch_3d(x_1,x_2,y,batchsize; shuffle=false,partial=false,x1type=typeof(x_1),x2type=typeof(x_2),ytype=typeof(y),xsize=size(x_1), ysize=size(y))
    nx = size(x_1)[end]
    if nx != size(y)[end]; throw(DimensionMismatch()); end
    x2_1 = reshape(x_1, :, nx)
    x2_2 = reshape(x_2, :, nx)
    y2 = reshape(y, :, nx)
    imax = partial ? nx : nx - batchsize + 1
    # xtype,ytype may be underspecified, here we infer the exact types from the first batch:
    ids = 1:min(nx,batchsize)
    xt_1 = typeof(convert(x1type, reshape(x2_1[:,ids],xsize[1:end-1]...,length(ids))))
    xt_2 = typeof(convert(x2type, reshape(x2_2[:,ids],xsize[1:end-1]...,length(ids))))
    yt = typeof(convert(ytype, reshape(y2[:,ids],ysize[1:end-1]...,length(ids))))
    Data3d{Tuple{xt_1,xt_2,yt}}(x2_1,x2_2,y2,batchsize,nx,partial,imax,1:nx,shuffle,xsize,ysize,x1type,x2type,ytype)
end

@propagate_inbounds function iterate(d::Data3d, i=0)     # returns data in d.indices[i+1:i+batchsize]
    if i >= d.imax
        return nothing
    end
    if d.shuffle && i == 0
        d.indices = randperm(d.length)
    end
    nexti = min(i + d.batchsize, d.length)
    ids = d.indices[i+1:nexti]
    x1batch = convert(d.x1type, reshape(d.x1[:,ids],d.xsize[1:end-1]...,length(ids)))
    x2batch = convert(d.x2type, reshape(d.x2[:,ids],d.xsize[1:end-1]...,length(ids)))
    ybatch = convert(d.ytype, reshape(d.y[:,ids],d.ysize[1:end-1]...,length(ids)))
    return ((x1batch,x2batch,ybatch),nexti)
end

eltype(::Type{Data3d{T}}) where T = T

function length(d::Data3d)
    n = d.length / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function rand(d::Data3d)
    i = rand(0:(d.length-d.batchsize))
    return iterate(d, i)[1]
end

# Use repeat(data,n) for multiple epochs; cycle(data) to go forever, take(data,n) for partial epochs

struct Repeat; data::Data3d; n::Int; end

repeat(d::Data3d, n::Int) = (@assert n >= 0; Repeat(d,n))
length(r::Repeat) = r.n * length(r.data)
eltype(r::Repeat) = eltype(r.data)
eltype(c::Cycle{Data3d}) = eltype(c.xs)
eltype(c::Cycle{Repeat}) = eltype(c.xs)

@propagate_inbounds function iterate(r::Repeat, s=(1,))
    epoch, state = s[1], tail(s)
    epoch > r.n && return nothing
    next = iterate(r.data, state...)
    next === nothing && return iterate(r, (epoch+1,))
    (next[1], (epoch, next[2]))
end

# Give length info in summary:
Base.summary(d::Data3d) = "$(length(d))-element $(typeof(d))"

