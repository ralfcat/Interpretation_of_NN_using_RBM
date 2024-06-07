using Flux
using Flux: onehotbatch, onecold, crossentropy
using Parameters
using Statistics

# Function to train a network with amino acid data
function train_network_AA(params::Hyperparameters, data::DataParameters, aminoAcidEncoding::AminoAcidEncoding)
    MSA = readdlm(data.file_name, ',')
    AA_dict = aminoAcidEncoding.AA_dict

    train_data, test_data = get_data_AA(MSA, AA_dict, 1)

    X_train = hcat([x[1] for x in train_data]...)
    y_train = onehotbatch([x[2] for x in train_data], 0:1)
    X_test = hcat([x[1] for x in test_data]...)
    y_test = onehotbatch([x[2] for x in test_data], 0:1)

    model = create_cnn(params, data)

    opt = Flux.ADAM(params.η)
    loss(x, y) = crossentropy(model(x), y)

    for epoch in 1:params.epochs
        Flux.train!(loss, Flux.params(model), [(X_train, y_train)], opt)
        println("Epoch $epoch: Train Loss = $(loss(X_train, y_train))")
    end

    train_accuracy = mean(onecold(model(X_train)) .== onecold(y_train))
    test_accuracy = mean(onecold(model(X_test)) .== onecold(y_test))

    return model, train_accuracy, loss(X_train, y_train), AA_dict, loss(X_test, y_test), test_accuracy
end


