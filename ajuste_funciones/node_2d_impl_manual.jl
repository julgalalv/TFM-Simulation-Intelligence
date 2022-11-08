using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots, LaTeXStrings

u0 = [2.0; 0.0]                 # valores iniciales
p_real = [-0.1  2.0 -2.0 -0.1]  # parametros reales de la ODE
p = [-0.9 1.0 -0.8 -0.3]          # parámetros iniciales
datasize = 60                   # tamaño de la muestra de datos
tspan = (0,1)               # intervalo ded tiempo
t = range(tspan[1],tspan[2],length=datasize)

n = 200                         # iteraciones de entrenamiento
s = (1000,800)                  # tamaño de las figuras

# ODE a aproximar
function ODEfunc(du,u,p,t)
    α, β, γ, δ = p  
    A = [α β; γ δ]
    du .= (*(A',(u.^3)))
end

prob = ODEProblem(ODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),p=p_real,saveat=t,reltol=1e-5,abstol=1e-5))  # datos de entrenamiento generados

# Gráfica del espacio de fases
plot(ode_data[1,:],ode_data[2,:],linewidth=3,c=:thermal, linez=t ,legend=false,xlims=(-3,3), ylims=(-2,2), colorbar = true,
    title = latexstring("u_1") * " vs " *latexstring("u_2") * " para " * latexstring("t") * " en " *  latexstring("[$(tspan[1]),$(tspan[2])]"))
xlabel!(latexstring("u_1"))
ylabel!(latexstring("u_2"))


params = Flux.params(p)   # parámetros del modelo

# Paso hacia adelante de la red
#=
function predict()
  solve(prob,Tsit5(),p=p,saveat=t,reltol=1e-5,abstol=1e-5)
end
=#
predict() = solve(prob,Tsit5(),p=p,saveat=t,reltol=1e-5,abstol=1e-5)

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
      println("Iteration $iter || Loss = $(loss())")
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
      tit = "NODE con " * latexstring("$(iter)") * " iteraciones"
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
