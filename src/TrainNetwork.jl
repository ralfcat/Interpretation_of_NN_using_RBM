using Flux
using Random
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, onehot
using NNlib
using DelimitedFiles
using Parameters


function training(MSA, AA_dict, params, data, device)
    # train network on the full input data
    @unpack lossFunction, η, optimizer, epochs = params
    @unpack file_name, input_length = data
    
    # encode data
    MSA = MSA[shuffle(1:end), :]
    data_encoded = MSA_encode(MSA, AA_dict)
    decision = MSA[1:end, end]
    train = [(data_encoded[i,:], decision[i]) for i in 1:length(decision)]
    
    # construct model
    model = create_perceptron(params, data) |> device

    # optimizer
    opt = Flux.setup(optimizer(η), model)
    
    # Training in epochs
    for epoch in 1:epochs
        for (x, y) in train
            # x, y = device(x), device(y)
            y = onehotbatch(y, 0:1)
            gs = gradient(m -> lossFunction(m(x), y), model)
            Flux.Optimise.update!(opt, model, gs[1])
        end
    end
    return model
end


# this function is to train the NN on discrete MSA data with CV
function train_network_AA(params, data, AminoAcidEncoding)
    # train network while performing CV, then on full data
    device = cpu
    @unpack lossFunction, η, optimizer, epochs, cv, seed, mode = params
    @unpack file_name, input_length = data

    Random.seed!(seed)
    
    # get data
    @unpack AA_dict = AminoAcidEncoding
    MSA = readdlm(file_name, ',')
    #shuffle MSA (very important, otherwise very bad results)
    MSA = MSA[shuffle(1:end), :]

    # initialize performance measures
    avg_acc = 0
    avg_loss = 0
    perf = [0, 0, 0, 0]
    loss = []
    weights = zeros(input_length, 2)

    for cross in 1:cv
        train_data, test_data = get_data_AA(MSA, AA_dict, cross, cv)
        # construct model
        model = create_perceptron(params, data) |> device

        # optimizer
        opt = Flux.setup(optimizer(η), model)
        
        # Training
        test_loss, test_acc = 0, 0
        for epoch in 1:epochs
            for (x, y) in train_data
                x, y = device(x), device(y)
                y = onehotbatch(y, 0:1)
                gs = gradient(m -> lossFunction(m(x), y), model)
                Flux.Optimise.update!(opt, model, gs[1])
            end
        end

        # evaluate performance for current fold
        test_loss, test_acc, perf_measure = loss_and_accuracy(test_data, model, device, params)
        println(" test_loss = $test_loss, test_accuracy = $test_acc, Performance: $perf_measure")
        
        # update average performance
        avg_acc += test_acc/cv
        avg_loss += test_loss/cv
        perf += perf_measure
    end

    println("Avg. accuracy: ", avg_acc, "\t avg. loss: $avg_loss", "\t Performance: $perf")
    
    # model trained on complete data
    model = training(MSA, AA_dict, params, data, device)

    # return model on complete data and further information
    return model, avg_acc, avg_loss, AA_dict, loss, weights
end
