using Zygote

## EJEMPLOS SIMPLES
# Gradiente de 3x^2 + 2x + 1^en x = 5
gradient(x -> 3x^2 + 2x + 1, 5)

# Gradiente de f(x,y) = xy en (2,3)
f(x,y) = x*y
gradient(f, [2, 3])

# Jacobiano de f(x,y) = [x^2,x*y+1] en (2,3)
jacobian((x,y) -> [x^2,x*y+1], 2, 3)

# Gradiente y Hessiano de f(x,y) = xy en (2,3)
f(x) = x[1]*x[2]
gradient(f, 2, 3)
hessian(f,[2,3])

## EJEMPLOS EN DISTINTAS ESTRUCTURAS
# Arrays (perceptrón)
W, b, x  = randn(2, 3), randn(2),  randn(3);

gradient((W,b) -> sum(W * x .+ b), W,b)

# Definición de potencia recursivamente con bucle for
function pow(x, n)
	r = 1
	for i = 1:n
	  r *= x
	end
	return r
  end

gradient(x -> pow(x, 3), 5)

# Definición de potencia recursivamente con if (op. ternario)
pow2(x, n) = n <= 0 ? 1 : x*pow2(x, n-1)

gradient(x -> pow2(x, 3), 5)

# En diccionarios
d = Dict()

gradient(5) do x 	# Equivalente a gradiente(x -> ..., 5)
	d[:x] = x		# asigna a la clave :x el valor x
	d[:x] * d[:x]	# devuelve el valor de la clave por sí mismo
end

d[:x] # El diccionario ha sido actualizado

## DESCENSO DEL GRADIENTE
# Definición capa densa
dense(W,b,σ = identity) = 
	x -> σ.(W * x .+ b)

# Composición de funciones
chain(f ...) = foldl(∘, reverse(f))

#Ejemplo de uso
chain(x -> x/2, x -> x + 1)(2) # (x -> x/2 + 1)(2) = 2

# Parámetros
W1,b1 = rand(5,10),rand(5)  # Capa 1
W2,b2 = randn(2,5),randn(2)	# Capa 2

# Perceptrón multicapa
mlp = chain(dense(W1,b1,tan),   # (10) -> (5)
            dense(W2,b2))       # (5)  -> (2)

x0 = randn(10) # valor inicial
mlp(x0)        # imagen del valor inicial

# loss: Error cuadrático con respecto a la constante 1 
loss(m,x) = sum(abs2,1 .- m(x)) 

_m, = gradient(mlp) do m # gradient(m -> loss(m),mlp)
	loss(m,x0)
end

# gradiente de loss respecto a W2
_m.outer.W


# ======== OTROS ==========
# Perceptrón multicapa a partir de lista de especificaciones
function create_mlp(layer_specs)
	layers = []
	foreach(x -> push!(layers,dense(x...)),layer_specs)
	chain(layers...)
end

layers = [(W1,b1,tan),(W2,b2)] # Especificaciones del modelo
mlp2 = create_mlp(layers) # Perceptrón multicapa
mlp2 == mlp