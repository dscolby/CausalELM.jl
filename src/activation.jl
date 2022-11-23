module ActivationFunctions

binarystep(x::Real) = ifelse(x < 0, 0, 1)

"""
    binarystep(x)

Apply the binary step activation function to an array or real number.

# Examples
```julia-repl
julia> binarystep(1)
1
julia> binarystep([-1000, 100, 1, 0, -0.001, -3])
[0, 1, 1, 1, 0, 0]
```
"""
binarystep(x::Array) = binarystep.(x)


@inline function σ(x::Real)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

"""
    σ(x)

Apply the sigmoid activation function to an array or real number.

# Examples
```julia-repl
julia> σ(1)
0.7310585786300049
julia> σ([1, 0])
[0.7310585786300049, 0.5]
```
"""
σ(x::Array) = σ.(x)

"""
    tanh(x)

Apply the tanh activation function to an array.

This is just a vectorized version of Base.tanh

# Examples
```julia-repl
julia> tanh([1, 0])
[0.7615941559557649, 0.0]
```
"""
tanh(x::Array) = Base.tanh.(x)

relu(x::Real) = ifelse(x < 0, zero(x), x)

"""
    relu(x)

Apply the ReLU activation function to an array or real number.

# Examples
```julia-repl
julia> relu(1)
1
julia> relu([1, 0, -1])
[1, 0, 0]
```
"""
relu(x::Array) = relu.(x)

leakyrelu(x::Real) = ifelse(x < 0, 0.01 * x, x)

"""
    relu(x)

Apply the leaky ReLU activation function to an array or real number.

# Examples
```julia-repl
julia> relu(1)
1
julia> relu([-0.01, 0, 1])
[1, 0, 0]
```
"""
leakyrelu(x::Array) = leakyrelu.(x)

swish(x::Real) = x * σ(x)

"""
    swish(x)

Apply the swish activation function to an array or real number.

# Examples
```julia-repl
julia> swish(1)
0.7310585786300049
julia> swish([1, 0, -1])
[0.7310585786300049, 0, -0.2689414213699951]
```
"""
swish(x::Array) = swish.(x)

softmax(x::Real) = exp(x) / sum(x)

"""
    swish(x)

Apply the softmax activation function to an array or real number.

For numbers that have large absolute values this function might be numerically unstable.

# Examples
```julia-repl
julia> softmax(1)
2.718281828459045
julia> softmax([1, -1])
[2.718281828459045, -0.36787944117144233]
```
"""
@inline function softmax(x::Array)
    z = max(x)
    
    if isfinite(z)
        @fastmath result = x .- max
    else
        @fastmath @. @fastmath @. result = ifelse(isequal(z,Inf), 
            ifelse(isequal(z, Inf), 1, 0), exp(x - max))
    end
    result ./= sum(intermediate)
end

softmax(x::Array) = softmax.(x)

softplus(x::Real) = log1p(exp(-abs(x))) + relu(x)

"""
    softplus(x)

Apply the softplus activation function to an array or real number.

For numbers that have large absolute values this function might be numerically unstable.

# Examples
```julia-repl
julia> softplus(1)
1.3132616875182228
julia> softplus([1, -1])
[1.3132616875182228, 0.31326168751822286]
```
"""
softplus(x::Array) = softplus.(x)

gelu(x::Real) = (x * (1 + Base.tanh(sqrt(2 / π) * (x + (0.044715 * x^3))))) / 2

"""
    gelu(x)

Apply the GeLU activation function to an array or real number.

# Examples
```julia-repl
julia> gelu(1)
0.8411919906082768
julia> gelu([-1, 0, 1])
[-0.15880800939172324, 0, 0.8411919906082768]
```
"""
gelu(x::Array) = gelu.(x)

gaussian(x::Real) = exp(-abs(x)^2)

"""
    gaussian(x)

Apply the gaussian activation function to an array or real number.

# Examples
```julia-repl
julia> gaussian(1)
0.11443511435028261
julia> gaussian([1, -1])
[0.36787944117144233, 0.36787944117144233]
```
"""
gaussian(x::Array) = gaussian.(x)

@inline function hardtanh(x::Real) 
    if x < -1
        -1
    elseif -1 <= x <= 1
        x
    else
        1
    end
end

"""
    hardtanh(x)

Apply the hardtanh activation function to an array or real number.

# Examples
```julia-repl
julia> hardtanh(-2)
-1
julia> gaussian([-2, 0, 2])
[-1, 0, 1]
```
"""
hardtanh(x::Array) = hardtanh.(x)

elish(x::Real) = ifelse(x >= 0, swish(x), ((exp(x)-1)) * σ(x))

"""
    elish(x)

Apply the ELiSH activation function to an array or real number.

# Examples
```julia-repl
julia> elish(1)
0.7310585786300049
julia> gaussian([-1, 1])
[-0.1700034015685479, 0.7310585786300049]
```
"""
elish(x::Array) = elish.(x)

end
