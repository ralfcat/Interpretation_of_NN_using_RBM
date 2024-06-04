# load necessarry packages and files
import Pkg; Pkg.add("Flux")
import Pkg; Pkg.add("Parameters")
import Pkg; Pkg.add("NNlib")

using DelimitedFiles
using Random
using Serialization
using Flux

include("DataPreparation.jl")
include("ModelDefinitions.jl")
include("Evaluation.jl")
include("Parameters.jl")
include("TrainNetwork.jl")

# set seed
Random.seed!(0)

# function to encode data, and run evaluateNetwork for all data points
function evaluateTestData(model, data_file, AA_dict)
    MSA = readdlm(data_file, ',')
    encoded_MSA = MSA_encode(MSA, AA_dict)

    measures = [0, 0, 0, 0]
    for i in 1:size(encoded_MSA)[1]
        pred = onecold(model(encoded_MSA[i,:]), 0:1)
        #if pred != MSA[i, end]
        #    println(model(encoded_MSA[i,:]), '\t', pred, '\t', MSA[i, end])
        #end
        measures += performance_measure(MSA[i, end], pred[1])
    end
    
    measures
end

# set parameters
# MTB data
#params = Hyperparameters(mode = "chain", actFunction = identity, η=0.0003, lossFunction = Flux.mse)
#data = DataParameters(file_name = "./data/Tubercolosis/Tb_Train1.csv", input_length = 28)

# AIV data
params = Hyperparameters(mode = "single", actFunction = Flux.celu, η=0.00005, lossFunction = Flux.logitbinarycrossentropy)
data = DataParameters(file_name = "./data/NS1/NS1_H5_H7_Train1.csv", input_length = 249)
aminoAcidEncoding = AminoAcidEncoding()




# train and evaluate network
m, acc, loss, dict, l, w = train_network_AA(params, data, aminoAcidEncoding)
serialize("model.dat", m)
serialize("dict.dat", dict)
# evaluate model on test data (and train 2)
val = evaluateTestData(m, "./data/NS1/NS1_H5_H7_Train2.csv", dict)
#val = evaluateTestData(m, "./data/Tubercolosis/Tb_Train2.csv", dict)

val2 = evaluateTestData(m, "./data/NS1/NS1_H5_H7_Test.csv", dict)
#val2 = evaluateTestData(m, "./data/Tubercolosis/Tb_Test.csv", dict)


