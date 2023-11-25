"""Activation functions for Extreme Learning machines"""
module ActivationFunctions

"""
    binary_step(x)

Apply the binary step activation function to a real number.

# Examples
```julia-repl
julia> binary_step(1)
1
```
"""
binary_step(x::Float64) = ifelse(x < 0, 0, 1)

"""
    binary_step(x)

Apply the binary step activation function to an array.

# Examples
```julia-repl
julia> binary_step([-1000, 100, 1, 0, -0.001, -3])
[0, 1, 1, 1, 0, 0]
```
"""
binary_step(x::Array{Float64}) = binary_step.(x)

"""
    σ(x)

Apply the sigmoid activation function to a real number.

# Examples
```julia-repl
julia> σ(1)
0.7310585786300049
```
"""
@inline function σ(x::Float64)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

"""
    σ(x)

Apply the sigmoid activation function to an array.

# Examples
```julia-repl
julia> σ([1, 0])
[0.7310585786300049, 0.5]
```
"""
σ(x::Array{Float64}) = σ.(x)

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
tanh(x::Array{Float64}) = @fastmath Base.tanh.(x)

"""
    relu(x)

Apply the ReLU activation function to a real number.

# Examples
```julia-repl
julia> relu(1)
1
```
"""
relu(x::Float64) = @fastmath ifelse(x < 0, zero(x), x)

"""
    relu(x)

Apply the ReLU activation function to an array.

# Examples
```julia-repl
julia> relu([1, 0, -1])
[1, 0, 0]
```
"""
relu(x::Array{Float64}) = relu.(x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to a real number.

# Examples
```julia-repl
julia> leaky_relu(1)
1
```
"""
leaky_relu(x::Float64) = @fastmath ifelse(x < 0, 0.01 * x, x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to an array.

# Examples
```julia-repl
julia> leaky_relu([-0.01, 0, 1])
[1, 0, 0]
```
"""
leaky_relu(x::Array{Float64}) = leaky_relu.(x)

"""
    swish(x)

Apply the swish activation function to a real number.

# Examples
```julia-repl
julia> swish(1)
0.7310585786300049
```
"""
swish(x::Float64) = x * σ(x)

"""
    swish(x)

Apply the swish activation function to an array.

# Examples
```julia-repl
julia> swish([1, 0, -1])
[0.7310585786300049, 0, -0.2689414213699951]
```
"""
swish(x::Array{Float64}) = swish.(x)

"""
    softmax(x)

Apply the softmax activation function to a real number.

For numbers that have large absolute values this function may become numerically unstable.

# Examples
```julia-repl
julia> softmax(1)
2.718281828459045
```
"""
softmax(x::Float64) = @fastmath exp(x) / sum(x)

"""
    softmax(x)

Apply the softmax activation function to an array.

For numbers that have large absolute values this function might be numerically unstable.

# Examples
```julia-repl
julia> softmax([1, -1])
[2.718281828459045, -0.36787944117144233]
```
"""
softmax(x::Array{Float64}) = softmax.(x)

"""
    softplus(x)

Apply the softplus activation function to a real number.

# Examples
```julia-repl
julia> softplus(1)
1.3132616875182228
```
"""
softplus(x::Float64) = @fastmath log1p(exp(-abs(x))) + relu(x)

"""
    softplus(x)

Apply the softplus activation function to an array.

# Examples
```julia-repl
julia> softplus([1, -1])
[1.3132616875182228, 0.31326168751822286]
```
"""
softplus(x::Array{Float64}) = softplus.(x)

"""
    gelu(x)

Apply the GeLU activation function to a real number.

# Examples
```julia-repl
julia> gelu(1)
0.8411919906082768
```
"""
gelu(x::Float64) = @fastmath (x * (1 + Base.tanh(sqrt(2 / π) * (x + (0.044715 * x^3))))) / 2

"""
    gelu(x)

Apply the GeLU activation function to an array.

# Examples
```julia-repl
julia> gelu([-1, 0, 1])
[-0.15880800939172324, 0, 0.8411919906082768]
```
"""
gelu(x::Array{Float64}) = gelu.(x)

"""
    gaussian(x)

Apply the gaussian activation function to a real number.

# Examples
```julia-repl
julia> gaussian(1)
0.11443511435028261
```
"""
gaussian(x::Float64) = @fastmath exp(-abs2(x))

"""
    gaussian(x)

Apply the gaussian activation function to an array.

# Examples
```julia-repl
julia> gaussian([1, -1])
[0.36787944117144233, 0.36787944117144233]
```
"""
gaussian(x::Array{Float64}) = gaussian.(x)

"""
    hard_tanh(x)

Apply the hard_tanh activation function to a real number.

# Examples
```julia-repl
julia> hard_tanh(-2)
-1
```
"""
@inline function hard_tanh(x::Float64) 
    if x < -1
        -1
    elseif -1 <= x <= 1
        x
    else
        1
    end
end

"""
    hard_tanh(x)

Apply the hard_tanh activation function to an array.

# Examples
```julia-repl
julia> hard_tanh([-2, 0, 2])
[-1, 0, 1]
```
"""
hard_tanh(x::Array{Float64}) = hard_tanh.(x)

"""
    elish(x)

Apply the ELiSH activation function to a real number.

# Examples
```julia-repl
julia> elish(1)
0.7310585786300049
```
"""
elish(x::Float64) = ifelse(x >= 0, swish(x), @fastmath ((exp(x)-1)) * σ(x))

"""
    elish(x)

Apply the ELiSH activation function to an array.

# Examples
```julia-repl
julia> elish([-1, 1])
[-0.17000340156854793, 0.7310585786300049]
```
"""
elish(x::Array{Float64}) = elish.(x)

"""
    fourrier(x)

Apply the Fourier activation function to a real number.

# Examples
```julia-repl
julia> fourier(1)
0.8414709848078965
```
"""
fourier(x::Float64) = @fastmath sin(x)

"""
    fourrier(x)

Apply the Fourier activation function to an array.

# Examples
```julia-repl
julia> fourier([-1, 1])
[-0.8414709848078965, 0.8414709848078965]
```
"""
fourier(x::Array{Float64}) = fourier.(x)

end
