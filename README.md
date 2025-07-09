<h1 align="center">HBSC</h1>


<p align="center">
<b>
An efficient vulnerability detection method for smart contracts.</b>

<p align="center">
Code release and supplementary materials for:</br>
<b>"Enhancing Smart Contract Vulnerability Detection via Dual-Source Feature Extraction and Fusion"</b></br>

## Repo structure
- `BasicBlock.py、BlockType.py`: Code for basic block dividing.   

- `Cfg.py、CfgBuilder.py`: Code for details of cfg generation.  

- `Generate CFG.py`: Code for generating cfg of bytecode. 

- `Semantic_feature.py`: Code for semantic features extraction of basic blocks.

- `Structural_feature.py`: Code for structural features extraction of ECFG.

- `get_rules`: Main objective functions extraction rules for vulnerabilities.

## Dependencies
- python>=3.8
- pyparsing=3.0.9
- pyevmasm=0.2.3
- numpy=1.24.3
- scipy=1.10.1
- scikit-learn=1.3.0
- solc-select=1.0.3
- torch=2.0.1

## How to Run:
### For source code:
- To generate objective functions set for vulnerabilities: get_rules folder
- To use Word2vec model to get embedding vectors.
### For bytecode:
- To divide basic blocks from bytecode: Basic block.py、BlockType.py
- To generate CFG of bytecode: Generate CFG.py
- To obtain semantic features of basic blocks: Semantic_feature.py
- To obtain structural features of ECFG: Structural_feature.py
  
