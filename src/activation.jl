"""
    binary_step(x)

Apply the binary step activation function to a real number.

# Examples
```julia
binary_step(1)
```
"""
binary_step(x) = ifelse(x < 0, 0, 1)

"""
    binary_step(x)

Apply the binary step activation function to an array.

# Examples
```julia
binary_step([-1000, 100, 1, 0, -0.001, -3])
```
"""
binary_step(x::Array{Float64}) = binary_step.(x)

"""
    σ(x)

Apply the sigmoid activation function to a real number.

# Examples
```julia
σ(1)
```
"""
@inline function σ(x)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

"""
    σ(x)

Apply the sigmoid activation function to an array.

# Examples
```julia
σ([1, 0])
```
"""
σ(x::Array{Float64}) = σ.(x)

"""
    tanh(x)

Apply the tanh activation function to an array.

# Notes
This is just a vectorized version of Base.tanh

# Examples
```julia
tanh([1, 0])
```
"""
Base.tanh(x::Array{Float64}) = @fastmath Base.tanh.(x)

"""
    relu(x)

Apply the ReLU activation function to a real number.

# Examples
```julia
relu(1)
```
"""
relu(x) = @fastmath ifelse(x < 0, zero(x), x)

"""
    relu(x)

Apply the ReLU activation function to an array.

# Examples
```julia
relu([1, 0, -1])
```
"""
relu(x::Array{Float64}) = relu.(x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to a real number.

# Examples
```julia
leaky_relu(1)
```
"""
leaky_relu(x) = @fastmath ifelse(x < 0, 0.01 * x, x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to an array.

# Examples
```julia
leaky_relu([-0.01, 0, 1])
```
"""
leaky_relu(x::Array{Float64}) = leaky_relu.(x)

"""
    swish(x)

Apply the swish activation function to a real number.

# Examples
```julia
swish(1)
```
"""
swish(x) = x * σ(x)

"""
    swish(x)

Apply the swish activation function to an array.

# Examples
```julia
swish([1, 0, -1])
```
"""
swish(x::Array{Float64}) = swish.(x)

"""
    softmax(x)

Apply the softmax activation function to a real number.

# Examples
```julia
softmax(1)
```
"""
softmax(x) = @fastmath exp(x) / sum(exp(x))

"""
    softmax(x)

Apply the softmax activation function to a vector.

# Examples
```julia
softmax([1, 2, 3])
```
"""
softmax(x::Vector{Float64}) = @fastmath exp.(x.-maximum(x))/sum(exp.(x.-maximum(x)))

"""
    softmax(x)

Apply the softmax activation function to the rows of an array.

# Examples
```julia
julia> x = rand(5, 3)
softmax(x)
```
"""
softmax(x::Array{Float64}) = mapslices(softmax, x, dims=2)

"""
    softplus(x)

Apply the softplus activation function to a real number.

# Examples
```julia
softplus(1)
```
"""
softplus(x) = @fastmath log1p(exp(-abs(x))) + relu(x)

"""
    softplus(x)

Apply the softplus activation function to an array.

# Examples
```julia
softplus([1, -1])
```
"""
softplus(x::Array{Float64}) = softplus.(x)

"""
    gelu(x)

Apply the GeLU activation function to a real number.

# Examples
```julia
gelu(1)
```
"""
gelu(x) = @fastmath (x * (1 + Base.tanh(sqrt(2 / π) * (x + (0.044715 * x^3))))) / 2

"""
    gelu(x)

Apply the GeLU activation function to an array.

# Examples
```julia
gelu([-1, 0, 1])
```
"""
gelu(x::Array{Float64}) = gelu.(x)

"""
    gaussian(x)

Apply the gaussian activation function to a real number.

# Examples
```julia
gaussian(1)
```
"""
gaussian(x) = @fastmath exp(-abs2(x))

"""
    gaussian(x)

Apply the gaussian activation function to an array.

# Examples
```julia
gaussian([1, -1])
```
"""
gaussian(x::Array{Float64}) = gaussian.(x)

"""
    hard_tanh(x)

Apply the hard_tanh activation function to a real number.

# Examples
```julia
hard_tanh(-2)
```
"""
@inline function hard_tanh(x) 
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
```julia
hard_tanh([-2, 0, 2])
```
"""
hard_tanh(x::Array{Float64}) = hard_tanh.(x)

"""
    elish(x)

Apply the ELiSH activation function to a real number.

# Examples
```julia
elish(1)
```
"""
elish(x) = ifelse(x >= 0, swish(x), @fastmath ((exp(x)-1)) * σ(x))

"""
    elish(x)

Apply the ELiSH activation function to an array.

# Examples
```julia
elish([-1, 1])
```
"""
elish(x::Array{Float64}) = elish.(x)

"""
    fourrier(x)

Apply the Fourier activation function to a real number.

# Examples
```julia
fourier(1)
```
"""
fourier(x) = @fastmath sin(x)

"""
    fourrier(x)

Apply the Fourier activation function to an array.

# Examples
```julia
fourier([-1, 1])
```
"""
fourier(x::Array{Float64}) = fourier.(x)
