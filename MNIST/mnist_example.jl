using DiffEqFlux, DifferentialEquations, Printf
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using MLDataUtils:  LabelEnc, convertlabel, stratifiedobs
using CUDA

# Evitamos el indexado escalar (empeora el rendimiento)
CUDA.allowscalar(false)

function loadmnist(batchsize = bs)
    # Conversión a OneHot
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # reshape de las imágenes y typo Float32
    reshape32(imgs) =  Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    
    # Cargamos los conjuntos de MNIST
    x_train, y_train = MLDatasets.MNIST.traindata()
    x_test, y_test = MLDatasets.MNIST.testdata()

    # Procesamiento de los datos 
    x_train_data =reshape32(x_train)
    y_train_data = onehot(y_train)
    x_test_data = reshape32(x_test)
    y_test_data = onehot(y_test)
        return (
        # Flux DataLoader para hacer minibatches y mezclar los datos
        DataLoader(gpu.(collect.((x_train_data, y_train_data))); batchsize = batchsize,
                   shuffle = true),
        DataLoader(gpu.(collect.((x_test_data, y_test_data))); batchsize = batchsize,
                   shuffle = false)
    )
end

# tamaño de batch
const bs = 128

# carga de datos
train_dataloader, test_dataloader = loadmnist(bs)

dim = 64                        # dimensión de capas ocultas
augmented = 0                   # dimensión aumentada de anode
augment_dim = dim + augmented   # dimensión total

# downsample de las imágenes a 6 x 6 x dim
down = Flux.Chain(Flux.Conv((3, 3), 1=>dim, relu, stride = 1), Flux.GroupNorm(dim, dim),
             Flux.Conv((4, 4), dim=>dim, relu, stride = 2, pad=1), Flux.GroupNorm(dim, dim),
             Flux.Conv((4, 4), dim=>dim, stride = 2, pad = 1)) |>gpu

# modelo de (A)NODE
dudt = Flux.Chain(Flux.Conv((3, 3), augment_dim=>augment_dim, tanh, stride=1, pad=1),
             Flux.Conv((3, 3), augment_dim=>augment_dim, tanh, stride=1, pad=1)) |>gpu

nn_ode = NeuralODE(dudt, (0.f0, 1.f0), Tsit5(),
                   save_everystep = false,
                   reltol = 1e-3, abstol = 1e-3,
                   save_start = false) |> gpu

# (A)NODE
node_model = augmented < 1 ? nn_ode : AugmentedNDELayer(nn_ode,augmented)             

# fully connected que devuelve la probabilidad de clase              
fc = Flux.Chain(Flux.GroupNorm(augment_dim, augment_dim), x -> relu.(x), Flux.MeanPool((6, 6)),
           x -> reshape(x, (augment_dim, :)), Flux.Dense(augment_dim,10),sigmoid) |> gpu
          
# conversión de DiffEqArray del solver ODE en una Matriz que puede ser utilizada en la siguiente capa
function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return xarr[:,:,:,:,1]
end

# Modelo
model = Flux.Chain(down,                # (28, 28, 1, BS) -> (6, 6, dim, BS)
              node_model,               # (6, 6, augment_dim, BS) -> (6, 6, augment_dim, BS, 1)
              DiffEqArray_to_Array,     # (6, 6, augment_dim, BS, 1) -> (6, 6, augment_dim, BS)
              fc)                       # (6, 6, augment_dim, BS) -> (10, BS)



# Clasificacion dado un vector de salida del modelo
classify(x) = argmax.(eachcol(x))

# Cálculo del rendimiento
function accuracy(model, data; n_batches = 100)
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(data)
        i > n_batches && break
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# Función de pérdida
loss(x, y) = logitcrossentropy(model(x), y)


# Ejemplo de paso hacia adelante con la primera imagen de entrenamiento
img, lab = train_dataloader.data[1][:, :, :, 1:1], train_dataloader.data[2][:, 1:1]
x_m = model(img)
# pérdida 
loss(img, lab)
# rendimiento inicial
accuracy(model, train_dataloader)

## Entrenamiento
opt = ADAM(0.05)
iter = 0

callback() = begin
    global iter += 1
    if iter % 10 == 1
        train_accuracy = accuracy(model, train_dataloader) * 100
        test_accuracy = accuracy(model, test_dataloader;
                                 n_batches = length(test_dataloader)) * 100
        @printf("Iter: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
                iter, train_accuracy, test_accuracy)
    end
end

Flux.train!(loss, Flux.params(down, nn_ode.p, fc), train_dataloader, opt, cb = callback)