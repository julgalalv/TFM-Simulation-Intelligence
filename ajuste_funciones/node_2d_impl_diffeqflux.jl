using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots, LaTeXStrings

u0 = [2.0; 0.0]               # valores iniciales
datasize = 60                 # tamaño de la muestra de datos
tspan = (0,1)                 # intervalo de tiempo
t = range(tspan[1],tspan[2],length=datasize)
input_dim, hidden_dim = 2, 64 # dimensiones de la nn
augment_dim = 2               # dimensiones aumentadas para ANODE
n = 200                       # iteraciones de entrenamiento
s = (1000,800)                # tamaño de las figuras

# Función que genera datos a partir de la ODE definida en ODEfunc()
function create_data()
  function ODEfunc(du,u,p,t)
      A = [-0.1  2.0; -2.0 -0.1]
      du .= (*(A',(u.^3)))
  end
  prob = ODEProblem(ODEfunc,u0,tspan)
  return Array(solve(prob,Tsit5(),saveat=t))
end

ode_data = create_data()  # datos de entrenamiento generados

# Función que construye una NODE o ANODE en función de los parámetros
# con el modelo definido
function construct_model(input_dim, hidden_dim, augment_dim)
  input_dim = input_dim + augment_dim
  dudt = Chain(x -> x.^3,
               Dense(input_dim,hidden_dim,tanh),
               Dense(hidden_dim,input_dim))
  n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-5,abstol=1e-5)
  model = augment_dim == 0 ? n_ode : AugmentedNDELayer(n_ode,augment_dim)
  return model
end


model = construct_model(input_dim, hidden_dim, augment_dim) # modelo
params = Flux.params(model)                                 # parámetros del modelo

# Paso hacia adelante de la red
function predict()
  model(u0)[1:2,:]
end

# Función de pérdida 
loss() = sum(abs2,ode_data .- predict())


data = Iterators.repeated((), n)  # iterador para el entrenamiento
opt = ADAM(0.1)                   # optimizador

loss_list = []
iter = 0
# Función callback para representar el entrenamiento
cb = function ()
  global iter
    if iter % 50 == 0
      pprintln("Iteration $iter || Loss = $(loss())")
      cur_pred = predict()
      x_cur, y_cur, x_data, y_data = cur_pred[1,:], cur_pred[2,:], ode_data[1,:], ode_data[2,:]
      plt = scatter(x_data,y_data,label="data",legend=:bottomright,xlims=(-3,3), ylims=(-2,2))
      plt = plot!(plt,x_cur,y_cur,linewidth=2,label="prediction",size=s, title = latexstring("u_1") * " vs " *latexstring("u_2"))
      xlabel!(latexstring("u_1"))
      ylabel!(latexstring("u_2"))
      plt_ = scatter(t,x_data,color = "green",label="data " * latexstring("u_2"), legend=:topright, size=s, xlims=(-0.1,tspan[2]+0.1), ylims=(-2.2,2.2))
      plot!(plt_,t,x_cur,color = "blue",linewidth=2,label="pred " * latexstring("u_2"), title = latexstring("u_1,u_2") * " vs " *latexstring("t"))
      xlabel!(latexstring("t"))
      ylabel!(latexstring("u_1, u_2"))
      scatter!(t,y_data,color = "yellow",label="data " * latexstring("u_2"))
      plot!(plt_,t,y_cur,color = "orange",linewidth=2,label="pred " * latexstring("u_2"))

      plt_loss = plot(loss_list,linewidth=2, legend = false,xlims=(0,n), ylims=(0,250),
                      size=s, annotations = (150,200,Plots.text("loss = $(loss())")),
                      title = "loss frente a iteraciones")
      xlabel!(latexstring("n"))
      ylabel!(latexstring("loss"))
      if augment_dim == 0
      tit = "NODE con " * latexstring("$(iter)") * " iteraciones"
      else
        tit = "ANODE(aumento = $(augment_dim)) con " * latexstring("$(iter)") * " iteraciones"
      end
      plt_tit = plot(title = tit, grid=false, showaxis = false, ticks = false)
      figure = plot(plt_tit,plt,plt_,plt_loss, layout = @layout([A{0.01h};[B C; D]]))
      display(figure)
    end
    append!(loss_list,loss()) 
    iter += 1
end

# Visualización con los valores iniciales
cb()

# Entrenamiento
Flux.train!(loss, params, data, opt, cb = cb)
