from gensim.models import Word2Vec
import numpy as np
import os
import glob
import re

seed = 42
np.random.seed(seed)

def process_blocks(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    blocks = []
    current_block = []
    for line in content:
        if line.strip() == '':
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line.strip())
    if current_block:
        blocks.append(current_block)

    tokenized_blocks = [[word for line in block for word in line.split()] for block in blocks]
    return tokenized_blocks

def train_word2vec_model(tokenized_blocks):
    model = Word2Vec(sentences=tokenized_blocks, vector_size=100, window=5, min_count=1, sg=1)
    return model.wv

def calculate_block_vectors(word_vectors, tokenized_blocks):
    block_vectors = []
    for block in tokenized_blocks:
        block_vector = sum(word_vectors[word] for word in block) / len(block)
        block_vectors.append(block_vector)
    return block_vectors

def modify_dot_file(dot_file_path, block_vectors):
    with open(dot_file_path, 'r') as file:
        dot_content = file.readlines()

    block_vector_index = 0
    modified_dot_content = []
    for line in dot_content:
        if 'label="<cfg BasicBlock@' in line and block_vector_index < len(block_vectors):
            vector_str = ', '.join(f"{x:.4f}" for x in block_vectors[block_vector_index][:10])
            line = line.split('label="')[0] + f'label="[ {vector_str} ]"]' + line.split(']')[-1]
            block_vector_index += 1
        modified_dot_content.append(line)

    with open(dot_file_path, 'w') as file:
        file.writelines(modified_dot_content)

def process_folders(dot_folder_path, txt_folder_path):
    dot_files = sorted(glob.glob(os.path.join(dot_folder_path, '*.dot')), key=lambda x: int(re.search(r'\d+', x).group()))
    txt_files = sorted(glob.glob(os.path.join(txt_folder_path, '*.txt')), key=lambda x: int(re.search(r'\d+', x).group()))

    for dot_file, txt_file in zip(dot_files, txt_files):
        tokenized_blocks = process_blocks(txt_file)
        word_vectors = train_word2vec_model(tokenized_blocks)
        block_vectors = calculate_block_vectors(word_vectors, tokenized_blocks)

