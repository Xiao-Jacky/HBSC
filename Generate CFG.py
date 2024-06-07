import os
from CfgBuilder import CfgBuilder
from Cfg import Cfg

def generate_cfg_from_bytecode(bytecode: str, contract_name: str) -> Cfg:
    builder = CfgBuilder()
    cfg = builder.buildCfg(contract_name, bytecode)
    return cfg

def process_folder(folder_path: str, output_folder: str):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".evm"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                bytecode = file.read()

            # 将文件名中的'.evm'替换为'.dot'
            dot_file_name = filename.replace('.evm', '.dot')
            dot_file_path = os.path.join(output_folder, dot_file_name)

            contract_name = filename.replace('.evm', '')
            cfg = generate_cfg_from_bytecode(bytecode, contract_name)

            # 保存为.dot文件
            cfg.storedot()

            print(f"CFG generated and saved for {contract_name}")

# 使用示例
source_folder = ""
output_folder = ""
process_folder(source_folder, output_folder)







