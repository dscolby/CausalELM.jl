"""
    binary_step(x)

Apply the binary step activation function.

# Examples
```jldoctest
julia> binary_step(1)
1

julia> binary_step([-1000, 100, 1, 0, -0.001, -3])
6-element Vector{Int64}:
 0
 1
 1
 1
 0
 0
```
"""
binary_step(x) = ifelse(x < 0, 0, 1)

binary_step(x::Array{Float64}) = binary_step.(x)

"""
    σ(x)

Apply the sigmoid activation function.

# Examples
```jldoctest
julia> σ(1)
0.7310585786300049

julia> σ([1.0, 0.0])
2-element Vector{Float64}:
 0.7310585786300049
 0.5
```
"""
@inline function σ(x)
    t = exp(-abs(x))
    return ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

σ(x::Array{Float64}) = σ.(x)

"""
    tanh(x)

Apply the tanh activation function.

# Examples
```jldoctest
julia> tanh([1.0, 0.0])
2-element Vector{Float64}:
 0.7615941559557649
 0.0
```
"""
Base.tanh(x::Array{Float64}) = @fastmath Base.tanh.(x)

"""
    relu(x)

Apply the ReLU activation function.

# Examples
```jldoctest
julia> relu(1)
1

julia> relu([1.0, 0.0, -1.0])
3-element Vector{Float64}:
 1.0
 0.0
 0.0
```
"""
relu(x) = @fastmath ifelse(x < 0, zero(x), x)

relu(x::Array{Float64}) = relu.(x)

"""
    leaky_relu(x)

Apply the leaky ReLU activation function to a number.

# Examples
```jldoctest
julia> leaky_relu(1)
1

julia> leaky_relu([-1.0, 0.0, 1.0])
3-element Vector{Float64}:
 -0.01
  0.0
  1.0
```
"""
leaky_relu(x) = @fastmath ifelse(x < 0, 0.01 * x, x)

leaky_relu(x::Array{Float64}) = leaky_relu.(x)

"""
    swish(x)

Apply the swish activation function to a number.

# Examples
```jldoctest
julia> swish(1)
0.7310585786300049

julia> swish([1.0, -1.0])
2-element Vector{Float64}:
  0.7310585786300049
 -0.2689414213699951
```
"""
swish(x) = x * σ(x)

swish(x::Array{Float64}) = swish.(x)

"""
    softmax(x)

Apply the softmax activation function to a number.

# Examples
```jldoctest
julia> softmax(1)
1.0

julia> softmax([1.0, 2.0, 3.0])
3-element Vector{Float64}:
 0.09003057317038045
 0.24472847105479764
 0.6652409557748219

julia> softmax([1.0 2.0 3.0; 4.0 5.0 6.0])
2×3 Matrix{Float64}:
 0.0900306  0.244728  0.665241
 0.0900306  0.244728  0.665241
```
"""
softmax(x) = @fastmath exp(x) / sum(exp(x))

softmax(x::Vector{Float64}) = @fastmath exp.(x .- maximum(x)) / sum(exp.(x .- maximum(x)))

softmax(x::Array{Float64}) = mapslices(softmax, x; dims=2)

"""
    softplus(x)

Apply the softplus activation function to a number.

# Examples
```jldoctest
julia> softplus(1)
1.3132616875182228

julia> softplus([1.0, -1.0])
2-element Vector{Float64}:
 1.3132616875182228
 0.3132616875182228
```
"""
softplus(x) = @fastmath log1p(exp(-abs(x))) + relu(x)

softplus(x::Array{Float64}) = softplus.(x)

"""
    gelu(x)

Apply the GeLU activation function to a number.

# Examples
```jldoctest
julia> gelu(1)
0.8411919906082768

julia> gelu([-1.0, 0.0])
2-element Vector{Float64}:
 -0.15880800939172324
  0.0
```
"""
gelu(x) = @fastmath (x * (1 + Base.tanh(sqrt(2 / π) * (x + (0.044715 * x^3))))) / 2

gelu(x::Array{Float64}) = gelu.(x)

"""
    gaussian(x)

Apply the gaussian activation function to a real number.

# Examples
```jldoctest
julia> gaussian(1)
0.36787944117144233

julia> gaussian([1.0, -1.0])
2-element Vector{Float64}:
 0.3678794411714423
 0.3678794411714423
```
"""
gaussian(x) = @fastmath exp(-abs2(x))

gaussian(x::Array{Float64}) = gaussian.(x)

"""
    hard_tanh(x)

Apply the hard_tanh activation function to a number.

# Examples
```jldoctest
julia> hard_tanh(-2)
-1

julia> hard_tanh([-2.0, 0.0, 2.0])
3-element Vector{Real}:
 -1
  0.0
  1
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

hard_tanh(x::Array{Float64}) = hard_tanh.(x)

"""
    elish(x)

Apply the ELiSH activation function to a number.

# Examples
```jldoctest
julia> elish(1)
0.7310585786300049

julia> elish([-1.0, 1.0])
2-element Vector{Float64}:
 -0.17000340156854793
  0.7310585786300049
```
"""
elish(x) = ifelse(x >= 0, swish(x), @fastmath ((exp(x) - 1)) * σ(x))

elish(x::Array{Float64}) = elish.(x)

"""
    fourrier(x)

Apply the Fourier activation function to a real number.

# Examples
```jldoctest
julia> fourier(1)
0.8414709848078965

julia> fourier([-1.0, 1.0])
2-element Vector{Float64}:
 -0.8414709848078965
  0.8414709848078965
```
"""
fourier(x) = @fastmath sin(x)

fourier(x::Array{Float64}) = fourier.(x)
