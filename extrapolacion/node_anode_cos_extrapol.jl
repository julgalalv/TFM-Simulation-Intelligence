using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots, LaTeXStrings


u0 = [1.0]                                              # valores iniciales u(0)
du0 = [0.0]                                             # valores iniciales du/dt(0)
tspan = (0.0,12)                                        # intervalo de tiempo de entrenamiento
tspan_pred = (tspan[1],tspan[2]+12)                     # intervalo extendido               
t = range(tspan[1],tspan[2],step=0.2)                   # rango de tiempo de entrenamiento
t_pred = range(tspan_pred[1],tspan_pred[2],step=0.2)    # rango de tiempo extendido
input_dim, hidden_dim = 1, 64                           # dimensiones de la nn
augment_dim = 1                                         # dimensiones aumentadas para ANODE
n = 400                                                 # iteraciones de entrenamiento
s = (1000,500)                                          # tamaño de las figuras

t_len = length(t)                                       # cantidad de datos de entrenamiento

# Función que genera datos a partir de la ODE definida en ODEfunc()
function create_data()
    function ODEfunc(ddu,du,u,p,t)
        ddu .= -u
    end
    prob = SecondOrderODEProblem(ODEfunc,u0,du0,tspan)
    return  Array(solve(prob,Tsit5(),saveat=t))[1,:]
  end

ode_data = create_data()  # datos de entrenamiento generados

# Función que construye una NODE o ANODE en función de los parámetros
# con el modelo definido
function construct_model(input_dim, hidden_dim, augment_dim)
    input_dim = augment_dim < 1 ? input_dim : input_dim + augment_dim
    chain = Chain(Dense(input_dim, hidden_dim, tanh),
                  Dense(hidden_dim, input_dim))
    n_ode = NeuralODE(chain ,tspan_pred, Tsit5(), saveat=t_pred,reltol=1e-5,abstol=1e-5) 
    model = augment_dim < 1 ? n_ode : AugmentedNDELayer(n_ode, augment_dim)
    return model, model.p             
end

model, parameters = construct_model(input_dim, hidden_dim, augment_dim) # modelo

# Paso hacia adelante de la red
function predict()
   model(u0)
end

# Función de pérdida 
loss() = sum(abs2,ode_data .- predict()[1,1:t_len]) 

data = Iterators.repeated((), n)  # iterador para el entrenamiento
opt = ADAM(0.005)                   # optimizador

loss_list = []
iter = 0
iter_threshold = 1
# Función callback para representar el entrenamiento
cb = function()
    global iter, iter_threshold
    l = loss()
    if iter % 1 == 0  
        println("Iteration $iter || Loss = $(l)")
        cur = predict()
        pl = scatter(t,ode_data, xlims=tspan_pred, ylims=(-1.20,1.20),label="data",legend=:topright, annotations = (6,-0.9,Plots.text("iter = $iter")))
        xlabel!(latexstring("t"))
        ylabel!(latexstring("u"))

        if augment_dim == 0
            tit = " - NODE"
            else
              tit = " -ANODE(aumento = $(augment_dim))"
            end

        plot!(pl,t_pred,cur[1,:],linewidth=2,label="prediction",title = latexstring("u") * " vs " *latexstring("t") * tit)

        plt_loss = plot(loss_list,linewidth=2, legend = false,xlims=(0,n), ylims=(0,50),
                      size=s, annotations = (150,30,Plots.text("loss = $(l)")),
                      title = "loss frente a iteraciones")
        xlabel!(latexstring("n"))
        ylabel!(latexstring("loss"))
        
        plt_tit = plot(title =tit, grid=false, showaxis = false, ticks = false)
        display(plot(pl,plt_loss, layout = @layout([A B])))
    end
    if l < 0.05 && iter > 1 && loss_list[iter-1] > 0.05 iter_threshold = iter end
    append!(loss_list,l) 
    iter += 1
end

# Visualización con los valores iniciales
cb()

# Entrenamiento
@time Flux.train!(loss, Flux.params(parameters, model), data, opt, cb = cb)
display(iter_threshold)



