# Unified Nonparametric Inference for Generalized Open Relation Extraction

This project provides tools for Unified Nonparametric Inference for Generalized Open Relation Extraction." 

Details about UNI-ORE: are in the paper and the implementation is based on the PyTorch library. 

## Quick Links
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Installation

For training, a GPU is recommended to accelerate the training speed. 

### PyTroch

The code is based on PyTorch 2.5.1. 

### Dependencies

The code is written in Python 3.10 

torch==2.5.1
numpy==1.26.4
scikit-learn==1.3.2
transformers==4.35.0

## Usage
* Run the full model on SemEval dataset with default hyperparameter settings<br>

```python src/main_UNI_ORE.py```<br>

## Acknowledgements
https://github.com/huggingface/transformers

https://github.com/qingwang-isu/MixORE