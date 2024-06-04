using DelimitedFiles
using Serialization
using Pkg
Pkg.add("Flux")
using Flux

include("DataPreparation.jl")
include("ModelDefinitions.jl")
include("Evaluation.jl")
include("Parameters.jl")
include("TrainNetwork.jl")

# after creating NN that in evaluateNetwork.jl saved as m
m = deserialize("model.dat")
dict = deserialize("dict.dat")

# choose data to label (AIV or MTB)
file_name = "./data/NS1/NS1_H5_H7_Train2.csv"
file_name = "./data/NS1/NS1_H5_H7_Test.csv"
#file_name = "./data/Tubercolosis/Tb_Train2.csv"
#file_name = "./data/Tubercolosis/Tb_Test.csv"

secTrain = readdlm(file_name, ',')



# function to encode data, and run evaluateNetwork for all data points
function evaluateTestData(model, MSA, AA_dict)
    encoded_MSA = MSA_encode(MSA, AA_dict)

    measures = [0, 0, 0, 0]
    p = []
    for i in 1:size(encoded_MSA)[1]
        pred = onecold(model(encoded_MSA[i,:]), 0:1)
        append!(p, pred)
        measures += performance_measure(MSA[i, end], pred)
    end
    
    return(p, measures)
end

pred, meas = evaluateTestData(m, secTrain, dict)
newLabel = pred

newData = hcat(secTrain, newLabel)

# choose data to label (AIV or MTB)

writedlm("./data/NS1/NewTrain2.csv", newData, ',')
writedlm("./data/NS1/NewTest.csv", newData, ',')

#writedlm("./data/Tubercolosis/NewTrain2.csv", newData, ',')
#writedlm("./data/Tubercolosis/NewTest.csv", newData, ',')
