using Parameters
using Flux

# Ensure these imports if Hyperparameters and DataParameters are defined in Parameters.jl
include("Parameters_CNN.jl")

# Function that makes a Convolutional Neural Network (CNN)
function create_cnn(params::Hyperparameters, data::DataParameters)
    @unpack input_length = data
    input_channels = 1
    encoded_length = input_length * 7
    
    conv1_output_size = encoded_length - 3 + 1
    pool1_output_size = div(conv1_output_size, 2)
    conv2_output_size = pool1_output_size - 3 + 1
    pool2_output_size = div(conv2_output_size, 2)
    conv3_output_size = pool2_output_size - 3 + 1
    pool3_output_size = div(conv3_output_size, 2)
    conv4_output_size = pool3_output_size - 3 + 1
    pool4_output_size = div(conv4_output_size, 2)
    conv5_output_size = pool4_output_size - 3 + 1
    pool5_output_size = div(conv5_output_size, 2)
    conv6_output_size = pool5_output_size - 3 + 1
    pool6_output_size = div(conv6_output_size, 2)
    conv7_output_size = pool6_output_size - 3 + 1
    pool7_output_size = div(conv7_output_size, 2)
    
    return Flux.Chain(
        x -> reshape(x, (encoded_length, 1, :)),  # Reshape input for convolution
        Conv((3,), 1 => 16, relu),                # 1D convolution
        MaxPool((2,)),                            # Max pooling
        Conv((3,), 16 => 32, relu),               # Another convolution layer
        MaxPool((2,)),                            # Another pooling layer
        Conv((3,), 32 => 64, relu),               # Another convolution layer
        MaxPool((2,)),                            # Another pooling layer
        Conv((3,), 64 => 128, relu),              # Another convolution layer
        MaxPool((2,)),                            # Another pooling layer
        Conv((3,), 128 => 256, relu),             # Another convolution layer
        MaxPool((2,)),                            # Another pooling layer
        Conv((3,), 256 => 512, relu),             # Another convolution layer
        MaxPool((2,)),                            # Another pooling layer
        Conv((3,), 512 => 1024, relu),            # Another convolution layer
        MaxPool((2,)),                            # Another pooling layer
        Flux.flatten,                             # Flatten for dense layers
        Dense(pool7_output_size * 1024, 128, relu), # Dense layer
        Dense(128, 2),                            # Output layer
        softmax                                   # Softmax for classification
    )
end

