import os
import safetensors.torch

def weight_quant(weight, num_bits=1):
    dtype = weight.dtype
    weight = weight.float()
    s = 1 / weight.abs().mean().clamp(min=1e-5)
    result = (weight * s).round().clamp(-1, 1)
    return result.type(dtype), s.item()

def generate_tensor_sizes_code(tensor_sizes):
    code = "// Generated code for tensor sizes\n"
    code += "#include <vector>\n"
    code += "#include <utility>\n\n"
    code += "std::vector<std::pair<size_t, size_t>> ModelWeightSizes = {\n"
    for input_size, output_size in tensor_sizes:
        code += f"    {{ {input_size}, {output_size} }},\n"
    code += "};\n"
    return code

def generate_cpp_files_from_safetensors(file_path, tensor_sizes):
    with open(file_path, 'rb') as file:
        data_bytes = file.read()
        data = safetensors.torch.load(data_bytes)

    for tensor_name, tensor_value in data.items():
        quantized_weight, scale_factor = weight_quant(tensor_value)
        if quantized_weight.dim() == 1:
            # For benchmarking we only process tensors with 2 dimensions
            continue
        if tensor_name == "model.embed_tokens.weight":
            # Ignore the embedding layer
            continue
        print(f"Processing tensor: {tensor_name} [input={quantized_weight.shape[1]} x output={quantized_weight.shape[0]}]")
        input_size = quantized_weight.shape[1]
        output_size = quantized_weight.shape[0]
        tensor_sizes.append((input_size, output_size))

    return tensor_sizes

if __name__ == "__main__":
    safetensors_dir = '../../bitnet_b1_58-3B/'
    output_dir = "generated_cpp_files"

    os.makedirs(output_dir, exist_ok=True)

    safetensors_files = [f for f in os.listdir(safetensors_dir) if f.endswith('.safetensors')]

    tensor_sizes = []
    for safetensors_file in safetensors_files:
        file_path = os.path.join(safetensors_dir, safetensors_file)
        tensor_sizes = generate_cpp_files_from_safetensors(file_path, tensor_sizes)

    tensor_sizes_code = generate_tensor_sizes_code(tensor_sizes)
    tensor_sizes_file = os.path.join(output_dir, "tensor_sizes.cpp")
    with open(tensor_sizes_file, 'w') as file:
        file.write(tensor_sizes_code)
    print(f"Generated C++ file for tensor sizes: {tensor_sizes_file}")
