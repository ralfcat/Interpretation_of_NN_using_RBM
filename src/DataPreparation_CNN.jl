#=
    This file contains function to prepare data for the use in Neural Networks.
    This includes split into test and validation set and encoding of non-numeric data.
=#

using Parameters
using Random

# function that returns sequence in numbers
function encode_seq(AA_dict, seq)
    encode = zeros(length(seq) * 7)
    for i in 1:length(seq)
        encode[((i-1)*7 + 1): (7*i)] = AA_dict[seq[i]]
    end
    encode
end

# function that returns MSA in numbers
function MSA_encode(MSA, AA_dict)
    data_encoded = zeros(Float64, size(MSA)[1], (size(MSA)[2]-1) * 7)
    for i in 1:size(MSA)[1]
        data_encoded[i,:] = encode_seq(AA_dict, MSA[i, 1:end-1])
    end
    return data_encoded
end


# get data for amino acid MSA
function get_data_AA(MSA, AA_dict, cross, cv=10)
    # MSA object should not have a header or rownames !
    data_encoded = MSA_encode(MSA, AA_dict)
    # data_encoded = undersample(MSA)
    decision = MSA[:, end]

    upper_test = min(trunc(Int, cross * round(length(decision)/cv)), size(data_encoded)[1])
    lower_test = (cross - 1) * trunc(Int, round(length(decision)/cv)) + 1

    test_data = [(data_encoded[i,:], decision[i]) for i in lower_test:upper_test-1]

    if lower_test == 1
        train_data = [(data_encoded[i,:], decision[i]) for i in upper_test:length(decision)]
    elseif upper_test == length(decision)
        train_data = [(data_encoded[i,:], decision[i]) for i in 1:lower_test-1]
        test_data = vcat(test_data, [(data_encoded[end,:], decision[end])])
    else
        train_data = [(data_encoded[i,:], decision[i]) for i in 1:lower_test-1]
        train_data = vcat(train_data, [(data_encoded[i,:], decision[i]) for i in upper_test:length(decision)])
    end
    
    return train_data, test_data

end


# get data for continuous data
function get_data(MSA)
    # data object should not have a header or rownames, but the decision !
    
    features = MSA[:,1:end-1]
    decision = MSA[1:end, end]

    return [(features[i,:], decision[i]) for i in 1:length(decision)]

end