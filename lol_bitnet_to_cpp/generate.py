import os
import safetensors.torch
import numpy as np

def weight_quant(weight,num_bits=1):
    dtype=weight.dtype
    weight=weight.float()
    s=1/weight.abs().mean().clamp(min=1e-5)
    result=(weight*s).round().clamp(-1,1)
    return result.type(dtype),s.item()

def generate_cpp_code(tensor_name,weight,scale_factor):
    if weight.dim() == 1:
        weight = weight.unsqueeze(0)
    input_size=weight.shape[1]
    output_size=weight.shape[0]

    code=f"// Generated code for tensor: {tensor_name} [input={input_size} x output={output_size}]\n"
    code+=f"#include <cstdint>\n\n"
    code+=f"#define X(index) static_cast<int16_t>( i[index] )\n"
    code+=f"void f{abs(hash(tensor_name))}(const int8_t*i, float*out) {{\n"  
    code+=f"const float S={scale_factor};\n"
    for output_index in range(output_size):
        nonzero = 0
        first = True
        for input_index in range(input_size):
            value=weight[output_index,input_index]
            if value==1:
                if nonzero > 0:
                    code += "+"
                else:
                    if first:
                        code+=f"out[{output_index}] = S*("
                        first = False
                    else:
                        code+=f"out[{output_index}] += S*("
                code += f"X({input_index})"
                nonzero += 1
            elif value==-1:
                if nonzero == 0:
                    if first:
                        code+=f"out[{output_index}] = S*("
                        first = False
                    else:
                        code+=f"out[{output_index}] += S*("
                code += "-"
                code += f"X({input_index})"
                nonzero += 1
            if nonzero >= 256:
                code += ");\n"
                nonzero = 0
        if nonzero > 0:
            code += ");\n"
    code+=f"}}\n"
    return code

def generate_cpp_files_from_safetensors(file_path):
    with open(file_path,'rb') as file:
        data_bytes=file.read()
    data=safetensors.torch.load(data_bytes)
    output_dir="generated_cpp_files"
    os.makedirs(output_dir,exist_ok=True)
    for tensor_name,tensor_value in data.items():
        quantized_weight,scale_factor=weight_quant(tensor_value)
        cpp_code=generate_cpp_code(tensor_name,quantized_weight,scale_factor)
        file_name=f"f{abs(hash(tensor_name)))}.cpp"
        file_path=os.path.join(output_dir,file_name)
        with open(file_path,'w') as file:
            file.write(cpp_code)
        print(f"Generated C++ file for tensor: {tensor_name}")

if __name__=="__main__":
    file_path='../bitnet_b1_58-3B/model-00001-of-00003.safetensors'
    generate_cpp_files_from_safetensors(file_path)
