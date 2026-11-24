# Finch.jl driver: @finch loop kernels, single-threaded.
#
#   julia finch_drv.jl PREFIX GROUP NITER_MAX
#
# GROUP: mttkrp16, mttkrp64, mttkrpfib16, mttkrpfib64, ttv, ttm16, ewise, sum. Reads PREFIX.tb
# (and PREFIX_y.tb, PREFIX_f<R>.f64, PREFIX_v.f64), as export_bin.py writes them. Finch is column-major: a tensor stored with
# axis o1 at the root, then o2, then o3 is indexed X[a, b, c] with a = o3 (leaf), b = o2,
# c = o1 (root). As for TACO, each kernel gets the mode order it wants: the output mode at the
# root for MTTKRP, the contracted or summed mode at the leaf for TTV, TTM and sums, the identity
# order for element-wise ops. Levels are Dense (root), then SparseList; the CSF arrays are
# built here from the sorted coordinates (prep_ms). The first call compiles the kernel
# (first_ms); then best of N. One JSON line per case, with a fingerprint as
# harness/common.py computes it.
using Finch


function read_tb(path)
    open(path) do f
        @assert String(read(f, 8)) == "TNSB0001"
        nmodes, nnz = read!(f, Vector{Int64}(undef, 2))
        dims = read!(f, Vector{Int64}(undef, nmodes))
        inds = [read!(f, Vector{Int64}(undef, nnz)) for _ in 1:nmodes]
        vals = read!(f, Vector{Float64}(undef, nnz))
        return inds, vals, dims
    end
end

read_f64(path, n) = open(f -> read!(f, Vector{Float64}(undef, n)), path)

# A Finch tensor of (inds, vals) with axis order[1] at the root (Dense) and the others as
# SparseList levels. Coordinates are 0-based in the file.
function build_csf(inds, vals, dims, order)
    n = length(order)
    nnz = length(vals)
    if order == collect(1:n)
        perm = collect(1:nnz)
    else
        key = zeros(UInt64, nnz)
        for ax in order
            d = UInt64(dims[ax])
            c = inds[ax]
            @inbounds for e in 1:nnz
                key[e] = key[e] * d + UInt64(c[e])
            end
        end
        perm = sortperm(key)
    end
    # compressed form at every level (0-based starts), as TACO's pos/crd
    pos = [Int[] for _ in 1:n]
    crd = [Int[] for _ in 1:n]
    val = Vector{Float64}(undef, nnz)
    @inbounds for q in 1:nnz
        e = perm[q]
        l = 1
        if q > 1
            p = perm[q-1]
            while l <= n && inds[order[l]][e] == inds[order[l]][p]
                l += 1
            end
        end
        for lev in l:n
            push!(crd[lev], inds[order[lev]][e])
            lev < n && push!(pos[lev+1], length(crd[lev+1]))
        end
        val[q] = vals[e]
    end
    for lev in 2:n
        push!(pos[lev], length(crd[lev]))
    end
    # root dense: level 2's pointer indexed by every root coordinate
    d0 = dims[order[1]]
    cnt = zeros(Int, d0)
    for (q, c) in enumerate(crd[1])
        cnt[c+1] = pos[2][q+1] - pos[2][q]
    end
    lvl = ElementLevel{0.0}(val)
    for lev in n:-1:2
        ptr = lev == 2 ? cumsum(vcat(1, cnt)) : pos[lev] .+ 1
        lvl = SparseListLevel{Int}(lvl, dims[order[lev]], ptr, crd[lev] .+ 1)
    end
    return Tensor(DenseLevel{Int}(lvl, d0))
end

dense_mat(M) = Tensor(Dense(Dense(Element(0.0))), M)   # M is R x n (column-major: r fastest)

# --- fingerprint (harness/common.py) -----------------------------------------------------
mutable struct FP
    count::UInt64
    h::UInt64
    v::Float64
    w::Float64
end
FP() = FP(0, 0, 0.0, 0.0)
function add!(fp::FP, lin::UInt64, val::Float64)
    val == 0 && return
    x = lin * 0x9E3779B97F4A7C15
    x ⊻= x >> 29
    x *= 0xBF58476D1CE4E5B9
    x ⊻= x >> 32
    fp.count += 1
    fp.h += x
    fp.v += val
    fp.w += val * (Float64(x >> 11) * 2.0^-53)
end

# Walk a Dense/SparseList/Element tree; coords arrive in Finch index order (leaf first)
walk(f, lvl::Finch.DenseLevel, pos, coords) =
    for i in 1:lvl.shape
        walk(f, lvl.lvl, (pos - 1) * lvl.shape + i, (i, coords...))
    end
walk(f, lvl::Finch.SparseListLevel, pos, coords) =
    for q in lvl.ptr[pos]:(lvl.ptr[pos+1]-1)
        walk(f, lvl.lvl, q, (lvl.idx[q], coords...))
    end
walk(f, lvl::Finch.ElementLevel, pos, coords) = f(coords, lvl.val[pos])

# `axes[t]` is the harness axis of Finch index t (0 for a factor's r); `shape` the harness
# output shape (with R last when there is an r index)
function fingerprint(T, axes, shape)
    fp = FP()
    nat = sortperm([a == 0 ? typemax(Int) : a for a in axes])   # Finch positions in harness order
    walk(T.lvl, 1, ()) do coords, v
        lin = UInt64(0)
        for (k, t) in enumerate(nat)
            lin = lin * UInt64(shape[k]) + UInt64(coords[t] - 1)
        end
        add!(fp, lin, v)
    end
    return "{\"shape\": [$(join(shape, ", "))], \"count\": $(fp.count), \"hsum\": \"0x$(string(fp.h, base=16))\", " *
           "\"vsum\": $(repr(fp.v)), \"wsum\": $(repr(fp.w)), \"sorted\": true}"
end

# --- kernels (3-d) ------------------------------------------------------------------------
function mttkrp!(A, X, Fb, Fa)
    @finch begin
        A .= 0
        for c = _, b = _, a = _, r = _
            A[r, c] += X[a, b, c] * Fb[r, b] * Fa[r, a]
        end
    end
    return A
end

# The SPLATT/CSF form: sum each (b, c) fiber against Fa into a workspace, then scale by Fb
function mttkrp_fib!(A, X, Fb, Fa, t)
    @finch begin
        A .= 0
        for c = _, b = _
            t .= 0
            for a = _, r = _
                t[r] += X[a, b, c] * Fa[r, a]
            end
            for r = _
                A[r, c] += t[r] * Fb[r, b]
            end
        end
    end
    return A
end

function ttv!(Y, X, v)
    @finch begin
        Y .= 0
        for c = _, b = _, a = _
            Y[b, c] += X[a, b, c] * v[a]
        end
    end
    return Y
end

function ttm!(Y, X, U)
    @finch begin
        Y .= 0
        for c = _, b = _, a = _, r = _
            Y[r, b, c] += X[a, b, c] * U[r, a]
        end
    end
    return Y
end

function sum3!(Y, X)
    @finch begin
        Y .= 0
        for c = _, b = _, a = _
            Y[b, c] += X[a, b, c]
        end
    end
    return Y
end

function add3!(Z, X, W)
    @finch begin
        Z .= 0
        for c = _, b = _, a = _
            Z[a, b, c] = X[a, b, c] + W[a, b, c]
        end
    end
    return Z
end

function mul3!(Z, X, W)
    @finch begin
        Z .= 0
        for c = _, b = _, a = _
            Z[a, b, c] = X[a, b, c] * W[a, b, c]
        end
    end
    return Z
end

function timeit(f, nmax)
    t0 = time_ns()
    f()
    first = (time_ns() - t0) / 1e9
    n = min(first < 0.1 ? 9 : first < 1 ? 5 : 3, nmax)
    ts = Float64[]
    for _ in 1:n
        GC.gc(false)
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    sort!(ts)
    return ts[1], ts[div(n, 2)+1], n, first
end

function emit(kase, t, prep, fp; variant = "")
    best, med, n, first = t
    kv = variant == "" ? "" : "\"kernel\": \"$variant\", "
    println("{\"lib\": \"finch\", $kv\"case\": \"$kase\", \"nthreads\": 1, \"best_ms\": $(1e3best), " *
            "\"median_ms\": $(1e3med), \"n\": $n, \"first_ms\": $(1e3first), \"prep_ms\": $(1e3prep), " *
            "\"version\": \"Finch $(pkgversion(Finch)), Julia $(VERSION)\", \"fp\": $fp}")
    flush(stdout)
end

function main(ds, group, nmax)
    inds, vals, dims = read_tb("$ds.tb")
    n = length(dims)
    @assert n == 3 "the Finch kernels here are 3-d"
    if group in ("mttkrp16", "mttkrp64", "ttm16", "mttkrpfib16", "mttkrpfib64")
        R = endswith(group, "64") ? 64 : 16
        fib = startswith(group, "mttkrpfib")
        flat = read_f64("$(ds)_f$R.f64", sum(dims) * R)
        F = Matrix{Float64}[]
        off = 0
        for d in dims   # file: d x R row-major == R x d column-major
            push!(F, reshape(flat[off+1:off+d*R], R, d))
            off += d * R
        end
        Ft = [dense_mat(M) for M in F]
        for m in 1:n
            others = [a for a in 1:n if a != m]
            if group == "ttm16"
                order = vcat(others, m)                   # m at the leaf: X[a=m, b=o2, c=o1]
                t0 = time_ns()
                X = build_csf(inds, vals, dims, order)
                prep = (time_ns() - t0) / 1e9
                Y = Tensor(Dense(SparseList(Dense(Element(0.0)))), R, dims[others[2]], dims[others[1]])
                t = timeit(() -> ttm!(Y, X, Ft[m]), nmax)
                fp = fingerprint(Y, [0, others[2], others[1]], [dims[others[1]], dims[others[2]], R])
                emit("ttm$(m-1)_r$R", t, prep, fp)
            else
                order = vcat(m, others)                   # m at the root: X[a=o3, b=o2, c=m]
                t0 = time_ns()
                X = build_csf(inds, vals, dims, order)
                prep = (time_ns() - t0) / 1e9
                A = Tensor(Dense(Dense(Element(0.0))), R, dims[m])
                if fib
                    w = Tensor(Dense(Element(0.0)), R)
                    t = timeit(() -> mttkrp_fib!(A, X, Ft[others[1]], Ft[others[2]], w), nmax)
                else
                    t = timeit(() -> mttkrp!(A, X, Ft[others[1]], Ft[others[2]]), nmax)
                end
                fp = fingerprint(A, [0, m], [dims[m], R])
                emit("mttkrp$(m-1)_r$R", t, prep, fp; variant = fib ? "fiber" : "plain")
            end
            X = nothing
            GC.gc()
        end
    elseif group in ("ttv", "sum")
        V = read_f64("$(ds)_v.f64", sum(dims))
        off = 0
        for m in 1:n
            others = [a for a in 1:n if a != m]
            order = vcat(others, m)
            t0 = time_ns()
            X = build_csf(inds, vals, dims, order)
            prep = (time_ns() - t0) / 1e9
            v = Tensor(Dense(Element(0.0)), V[off+1:off+dims[m]])
            off += dims[m]
            Y = Tensor(Dense(SparseList(Element(0.0))), dims[others[2]], dims[others[1]])
            if group == "ttv"
                t = timeit(() -> ttv!(Y, X, v), nmax)
            else
                t = timeit(() -> sum3!(Y, X), nmax)
            end
            fp = fingerprint(Y, [others[2], others[1]], [dims[others[1]], dims[others[2]]])
            emit("$(group)$(m-1)", t, prep, fp)
            X = nothing
            GC.gc()
        end
    elseif group == "ewise"
        yinds, yvals, _ = read_tb("$(ds)_y.tb")
        t0 = time_ns()
        X = build_csf(inds, vals, dims, [1, 2, 3])
        W = build_csf(yinds, yvals, dims, [1, 2, 3])
        prep = (time_ns() - t0) / 1e9
        for (kase, k!) in (("add_same", add3!), ("mul_same", mul3!))
            Z = Tensor(Dense(SparseList(SparseList(Element(0.0)))), dims[3], dims[2], dims[1])
            t = timeit(() -> k!(Z, X, W), nmax)
            emit(kase, t, prep, fingerprint(Z, [3, 2, 1], dims))
            Z = nothing
            GC.gc()
        end
    else
        error("unknown group $group")
    end
end

main(ARGS[1], ARGS[2], parse(Int, ARGS[3]))
