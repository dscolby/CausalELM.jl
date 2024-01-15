"""
    binary_step(x)

Apply the binary step activation function to a real number.

# Examples
```jldoctest
julia> binary_step(1)
 1.0
```
"""
binary_step(x) = ifelse(x < 0, 0, 1)

"""
    binary_step(x)

Apply the binary step activation function to an array.

# Examples
```jldoctest
julia> binary_step([-1000, 100, 1, 0, -0.001, -3])
6-element Vector{Float64}
 0.0 
 1.0 
 1.0 
 1.0 
 0.0 
 0.0
```
"""
binary_step(x::Array{Float64}) = binary_step.(x)

"""
    σ(x)

Apply the sigmoid activation function to a real number.

# Examples
```jldoctest
julia> σ(1)
 0.7310585786300049
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
```jldoctest
julia> σ([1, 0])
2-element Vector{Float64}
 0.7310585786300049
 0.5
```
"""
σ(x::Array{Float64}) = σ.(x)

"""
    tanh(x)

Apply the tanh activation function to an array.

This is just a vectorized version of Base.tanh

# Examples
```jldoctest
julia> tanh([1, 0])
2-element Vector{Float64}
 0.7615941559557649 
 0.0
```
"""
Base.tanh(x::Array{Float64}) = @fastmath Base.tanh.(x)

"""
    relu(x)

Apply the ReLU activation function to a real number.

# Examples
```jldoctest
julia> relu(1)
 1.0
```
"""
relu(x) = @fastmath ifelse(x < 0, zero(x), x)

"""
    relu(x)

Apply the ReLU activation function to an array.

# Examples
```jldoctest
julia> relu([1, 0, -1])
3-element Vector{Float64}
 1.0 
 0.0 
 0.0
```
"""
relu(x::Array{Float64}) = relu.(x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to a real number.

# Examples
```jldoctest
julia> leaky_relu(1)
 1.0
```
"""
leaky_relu(x) = @fastmath ifelse(x < 0, 0.01 * x, x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to an array.

# Examples
```jldoctest
julia> leaky_relu([-0.01, 0, 1])
3-element Vector{Float64}
 1.0 
 0.0 
 0.0
```
"""
leaky_relu(x::Array{Float64}) = leaky_relu.(x)

"""
    swish(x)

Apply the swish activation function to a real number.

# Examples
```jldoctest
julia> swish(1)
 0.7310585786300049
```
"""
swish(x) = x * σ(x)

"""
    swish(x)

Apply the swish activation function to an array.

# Examples
```jldoctest
julia> swish([1, 0, -1])
3-element Vector{Float64}
 0.7310585786300049 
 0.0 
 -0.2689414213699951
```
"""
swish(x::Array{Float64}) = swish.(x)

"""
    softmax(x)

Apply the softmax activation function to a real number.

# Examples
```jldoctest
julia> softmax(1)
 2.718281828459045
```
"""
softmax(x) = @fastmath exp(x) / sum(exp(x))

"""
    softmax(x)

Apply the softmax activation function to a vector.

# Examples
```jldoctest
julia> softmax([1, 2, 3])
3-element Vector{Float64}:
 0.09003057317038046
 0.24472847105479767
 0.6652409557748219
```
"""
softmax(x::Vector{Float64}) = @fastmath exp.(x)/sum(exp.(x))

"""
    softmax(x)

Apply the softmax activation function to the rows of an array.

# Examples
```jldoctest
julia> x = rand(5, 3)
5×3 Matrix{Float64}:
 0.482117  0.225359  0.615589
 0.255572  0.165051  0.427035
 0.387384  0.424856  0.369219
 0.175362  0.172561  0.111878
 0.508207  0.258347  0.591111
julia> softmax(x)
5×3 Matrix{Float64}:
 0.342895  0.265248  0.391857
 0.322529  0.294616  0.382855
 0.331106  0.343749  0.325146
 0.340635  0.339682  0.319682
 0.348998  0.271838  0.379164
```
"""
softmax(x::Array{Float64}) = mapslices(softmax, x, dims=2)

"""
    softplus(x)

Apply the softplus activation function to a real number.

# Examples
```jldoctest
julia> softplus(1)
 1.3132616875182228
```
"""
softplus(x) = @fastmath log1p(exp(-abs(x))) + relu(x)

"""
    softplus(x)

Apply the softplus activation function to an array.

# Examples
```jldoctest
julia> softplus([1, -1])
2-element Vector{Float64}
 1.3132616875182228 
 0.31326168751822286
```
"""
softplus(x::Array{Float64}) = softplus.(x)

"""
    gelu(x)

Apply the GeLU activation function to a real number.

# Examples
```jldoctest
julia> gelu(1)
 0.8411919906082768
```
"""
gelu(x) = @fastmath (x * (1 + Base.tanh(sqrt(2 / π) * (x + (0.044715 * x^3))))) / 2

"""
    gelu(x)

Apply the GeLU activation function to an array.

# Examples
```jldoctest
julia> gelu([-1, 0, 1])
3-element Vector{Float64}
 -0.15880800939172324 
 0.0 
 0.8411919906082768
```
"""
gelu(x::Array{Float64}) = gelu.(x)

"""
    gaussian(x)

Apply the gaussian activation function to a real number.

# Examples
```jldoctest
julia> gaussian(1)
 0.11443511435028261
```
"""
gaussian(x) = @fastmath exp(-abs2(x))

"""
    gaussian(x)

Apply the gaussian activation function to an array.

# Examples
```jldoctest
julia> gaussian([1, -1])
2-element Vector{Float64}
 0.36787944117144233 
 0.36787944117144233
```
"""
gaussian(x::Array{Float64}) = gaussian.(x)

"""
    hard_tanh(x)

Apply the hard_tanh activation function to a real number.

# Examples
```jldoctest
julia> hard_tanh(-2)
 -1.0
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
```jldoctest
julia> hard_tanh([-2, 0, 2])
3-element Vector{Float64}
 -1.0 
 0.0 
 1.0
```
"""
hard_tanh(x::Array{Float64}) = hard_tanh.(x)

"""
    elish(x)

Apply the ELiSH activation function to a real number.

# Examples
```jldoctest
julia> elish(1)
 0.7310585786300049
```
"""
elish(x) = ifelse(x >= 0, swish(x), @fastmath ((exp(x)-1)) * σ(x))

"""
    elish(x)

Apply the ELiSH activation function to an array.

# Examples
```jldoctest
julia> elish([-1, 1])
2-element Vector{Float64}
 -0.17000340156854793 
 0.7310585786300049
```
"""
elish(x::Array{Float64}) = elish.(x)

"""
    fourrier(x)

Apply the Fourier activation function to a real number.

# Examples
```jldoctest
julia> fourier(1)
 0.8414709848078965
```
"""
fourier(x) = @fastmath sin(x)

"""
    fourrier(x)

Apply the Fourier activation function to an array.

# Examples
```jldoctest
julia> fourier([-1, 1])
2-element Vector{Float64}
 -0.8414709848078965 
 0.8414709848078965
```
"""
fourier(x::Array{Float64}) = fourier.(x)
