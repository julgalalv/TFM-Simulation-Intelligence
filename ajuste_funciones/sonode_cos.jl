using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
using Flux.Data: DataLoader

# consideramos el estado aumentado h(t) = [u(t) a(t)]
u0 = [1.0]                                      # valor inicial de u(0)
du0 = [0.0]                                     # valor inicial de du/dt(0)
α0 = [0]                                        # valor inicial de a(0)
p = [0.0 0.0 0.0]                               # parámetros iniciales 
datasize = 50                                   # tamaño de la muestra de datos
tspan = (0.0,12.0)                              # intervalo de tiempo
t = range(tspan[1],tspan[2],length=datasize)
n = 251                                         # iteraciones de entrenamiento
s = (1000,500)                                  # tamaño de las figuras

# Función que genera datos a partir de la ODE definida en ODEfunc()
function create_data()
    function trueODEfunc(ddu,du,u,p,t)
        ddu .= -u
    end
    prob = SecondOrderODEProblem(trueODEfunc,u0,du0,tspan)
    return  Array(solve(prob,Tsit5(),saveat=t))[1,:]
end

ode_data = create_data()

# Modelo de la solución (SONODE como anode con restricciones)
function ODEsystem(dh,h,p,t)
    u, α = h
    β, δ, γ = p
    dh[1] = du = α
    dh[2] = dα = -β*u + δ*α + γ * u * α
end

# Perceptrón multicapa para inferir a(0)
hidden_dim = 64
nn_vel0 = Chain(Dense(1,hidden_dim,relu),Dense(hidden_dim,1))

#Parámetros del modelo
params = Flux.params(p, nn_vel0)

# Paso hacia adelante de la red
function predict() 
    h0 = [u0[1], nn_vel0(α0)[1]]  # estado inicial dependiente del perceptrón multicapa
    prob = ODEProblem(ODEsystem,h0,tspan,p)
    solve(prob,Tsit5(),p=p,saveat=t) 
end

# Función de pérdida
loss() = sum(abs2,ode_data .- predict()[1,:]) 

data = Iterators.repeated((), n)  # iterador para el entrenamiento
opt = ADAM(0.1)                   # optimizador

loss_list = []
iter = 0
iter_threshold = 1
# Función callback para representar el entrenamient
cb = function()
    global iter, iter_threshold
    l = loss()
    if iter % 50 == 0
       println("Iteration $iter || Loss = $(l)")
        cur = predict()
        cur_pred = cur[1,:]
        cur_vel = cur[2,:]
        pl = scatter(t,ode_data, xlims=tspan, ylims=(-1.20,1.20),label="data",  annotations = (6,-0.9,Plots.text("iter = $iter")))
        pl =plot(pl,t,cur_pred,linewidth=2,label="prediction",title = latexstring("u") * " vs " * latexstring("t") * " - SONODE")
        xlabel!(latexstring("t"))
        ylabel!(latexstring("u"))
        plt_loss = plot(loss_list,linewidth=2, legend = false,xlims=(0,n), ylims=(0,50),
        size=s, annotations = (150,30,Plots.text("loss = $(l)")),
        title = "loss frente a iteraciones")
        xlabel!(latexstring("n"))
        ylabel!(latexstring("loss"))
        display(plot(pl,plt_loss, layout = @layout([A B])))
    end
    if l < 0.05 && iter > 1 && loss_list[iter-1] > 0.05 iter_threshold = iter end
    append!(loss_list,l) 
    iter += 1
end



time = @time Flux.train!(loss, params, data, opt, cb = cb)
display(iter_threshold)



