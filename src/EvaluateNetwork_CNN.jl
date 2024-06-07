# Load necessary packages and files
import Pkg
Pkg.add("Flux")
Pkg.add("Parameters")
Pkg.add("NNlib")

using DelimitedFiles
using Random
using Serialization
using Flux

include("DataPreparation_CNN.jl")
include("ModelDefinitions_CNN.jl")
include("Evaluation.jl")
include("Parameters_CNN.jl")
include("train_network_AA_CNN.jl")

# Set seed
Random.seed!(0)

# Function to encode data, and run evaluateNetwork for all data points
function evaluateTestData(model, data_file, AA_dict)
    MSA = readdlm(data_file, ',')
    encoded_MSA = MSA_encode(MSA, AA_dict)
    correct_predictions = 0
    total_predictions = size(encoded_MSA)[1]

    measures = [0, 0, 0, 0]
    for i in 1:total_predictions
        pred = onecold(model(encoded_MSA[i,:]), 0:1)
        if pred[1] == MSA[i, end]
            correct_predictions += 1
        end
        measures += performance_measure(MSA[i, end], pred[1])
    end
    
    accuracy = correct_predictions / total_predictions
    println("Accuracy: ", accuracy)
    
    measures
end

# Set parameters
# MTB data
# params = Hyperparameters(mode = "chain", actFunction = identity, η=0.0003, lossFunction = Flux.mse)
# data = DataParameters(file_name = "./data/Tubercolosis/Tb_Train1.csv", input_length = 28)

# AIV data
params = Hyperparameters(mode = "single", actFunction = Flux.celu, η=0.00005, lossFunction = Flux.logitbinarycrossentropy)
data = DataParameters(file_name = "./data/NS1/NS1_H5_H7_Train1.csv", input_length = 249)
aminoAcidEncoding = AminoAcidEncoding()

# Train and evaluate network
m, acc, loss, dict, l, w = train_network_AA(params, data, aminoAcidEncoding)
serialize("model.dat", m)
serialize("dict.dat", dict)

# Print the training accuracy
println("Training Accuracy: ", acc)

# Evaluate model on test data (and train 2)
println("Validation Results on Train2:")
val = evaluateTestData(m, "./data/NS1/NS1_H5_H7_Train2.csv", dict)
# val = evaluateTestData(m, "./data/Tubercolosis/Tb_Train2.csv", dict)

println("Validation Results on Test:")
val2 = evaluateTestData(m, "./data/NS1/NS1_H5_H7_Test.csv", dict)
# val2 = evaluateTestData(m, "./data/Tubercolosis/Tb_Test.csv", dict)