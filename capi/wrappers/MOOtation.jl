# SPDX-License-Identifier: Apache-2.0
#
# MOOtation from Julia, through the C ABI.
#
#     include("MOOtation.jl")
#     using .MOOtation
#
#     zdt1(x) = begin
#         g = 1 + 9 * sum(x[2:end]) / (length(x) - 1)
#         [x[1], g * (1 - sqrt(x[1] / g))]
#     end
#
#     X, F = MOOtation.minimize(zdt1, """
#         algorithm = nsga2
#         pop_size  = 40
#         max_gen   = 100
#         n_vars    = 10
#         n_objs    = 2
#         lower     = 0
#         upper     = 1
#     """; library = "libmootation.so")
#
# There is nothing to build: this calls the same `extern "C"` entry points as
# every other language binding. Copy the file into your project.
#
# Julia is 1-indexed and column-major; the ABI is 0-indexed and row-major, so
# the reshapes below transpose deliberately. Getting that backwards silently
# scrambles decision vectors, which is why it is done in one place.

module MOOtation

export minimize, ask_tell, algorithms, version

const LIB = Ref{String}("libmootation")

setlibrary(path::AbstractString) = (LIB[] = path)

# ── Raw entry points ────────────────────────────────────────────────────────

version() = unsafe_string(@ccall LIB[].moo_version_string()::Cstring)

function algorithms()
    n = @ccall LIB[].moo_algorithm_count()::Cint
    n < 0 && error("moo_algorithm_count failed")
    [unsafe_string(@ccall LIB[].moo_algorithm_name(i::Cint)::Cstring)
     for i in 0:(n - 1)]
end

mutable struct Session
    handle::Ptr{Cvoid}
    n_vars::Int
    n_objs::Int
    n_cons::Int
end

function last_error(handle::Ptr{Cvoid})
    msg = @ccall LIB[].moo_last_error(handle::Ptr{Cvoid})::Cstring
    msg == C_NULL ? "unknown error" : unsafe_string(msg)
end

function open_session(settings::AbstractString)
    h = @ccall LIB[].moo_open(settings::Cstring)::Ptr{Cvoid}
    h == C_NULL && error("moo_open: " * last_error(C_NULL))
    Session(h,
            @ccall(LIB[].moo_n_vars(h::Ptr{Cvoid})::Cint),
            @ccall(LIB[].moo_n_objs(h::Ptr{Cvoid})::Cint),
            @ccall(LIB[].moo_n_cons(h::Ptr{Cvoid})::Cint))
end

close_session(s::Session) =
    (@ccall LIB[].moo_close(s.handle::Ptr{Cvoid})::Cvoid; s.handle = C_NULL)

check(s::Session, v::Integer) =
    v < 0 ? error(last_error(s.handle)) : v

# ── ask / tell ──────────────────────────────────────────────────────────────

"""
    ask(s) -> Matrix{Float64}   (n_candidates x n_vars)

Empty when the run has finished. The row count is the ALGORITHM's choice, not
`pop_size`: generational algorithms hand over a whole offspring generation,
steady-state ones one candidate at a time.
"""
function ask(s::Session)
    n = check(s, @ccall LIB[].moo_ask_count(s.handle::Ptr{Cvoid})::Cint)
    n == 0 && return zeros(Float64, 0, s.n_vars)
    buf = Vector{Float64}(undef, n * s.n_vars)
    check(s, @ccall LIB[].moo_ask(s.handle::Ptr{Cvoid}, buf::Ptr{Float64},
                                  Cint(length(buf))::Cint)::Cint)
    # The ABI writes row-major; reshape fills column-major, so build the
    # transpose and flip it back.
    permutedims(reshape(buf, s.n_vars, n))
end

"""
    tell(s, F [, G]) -> Int

`F` is (n_candidates x n_objs). Returns the size of the NEXT batch, 0 when the
run is over.
"""
function tell(s::Session, F::AbstractMatrix, G = nothing)
    fbuf = vec(permutedims(Float64.(F)))
    if s.n_cons > 0
        G === nothing && error("this run has $(s.n_cons) constraints; pass G")
        gbuf = vec(permutedims(Float64.(G)))
        return check(s, @ccall LIB[].moo_tell(s.handle::Ptr{Cvoid},
                                              fbuf::Ptr{Float64}, Cint(length(fbuf))::Cint,
                                              gbuf::Ptr{Float64}, Cint(length(gbuf))::Cint)::Cint)
    end
    check(s, @ccall LIB[].moo_tell(s.handle::Ptr{Cvoid},
                                   fbuf::Ptr{Float64}, Cint(length(fbuf))::Cint,
                                   C_NULL::Ptr{Float64}, Cint(0)::Cint)::Cint)
end

"""
    result(s) -> (X, F, cv)

The final population: X is (n x n_vars), F is (n x n_objs), cv is length n.
"""
function result(s::Session)
    n = check(s, @ccall LIB[].moo_result_count(s.handle::Ptr{Cvoid})::Cint)
    xb = Vector{Float64}(undef, n * s.n_vars)
    fb = Vector{Float64}(undef, n * s.n_objs)
    cb = Vector{Float64}(undef, n)
    check(s, @ccall LIB[].moo_result(s.handle::Ptr{Cvoid},
                                     xb::Ptr{Float64}, Cint(length(xb))::Cint,
                                     fb::Ptr{Float64}, Cint(length(fb))::Cint,
                                     cb::Ptr{Float64}, Cint(n)::Cint)::Cint)
    (permutedims(reshape(xb, s.n_vars, n)),
     permutedims(reshape(fb, s.n_objs, n)),
     cb)
end

# ── The convenient form ─────────────────────────────────────────────────────

"""
    minimize(f, settings; library, constraints=nothing) -> (X, F, cv)

`f` takes one decision vector and returns its objective values. `settings` is
the `key = value` text of mootation/settings.hpp.
"""
function minimize(f, settings::AbstractString;
                  library::Union{Nothing,AbstractString} = nothing,
                  constraints = nothing)
    library !== nothing && setlibrary(library)
    s = open_session(settings)
    try
        n = check(s, @ccall LIB[].moo_ask_count(s.handle::Ptr{Cvoid})::Cint)
        while n > 0
            X = ask(s)
            F = reduce(vcat, (reshape(Float64.(f(X[i, :])), 1, :)
                              for i in 1:size(X, 1)))
            if constraints === nothing
                n = tell(s, F)
            else
                G = reduce(vcat, (reshape(Float64.(constraints(X[i, :])), 1, :)
                                  for i in 1:size(X, 1)))
                n = tell(s, F, G)
            end
        end
        return result(s)
    finally
        close_session(s)
    end
end

end # module
