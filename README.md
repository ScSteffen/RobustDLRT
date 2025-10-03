# Robust and efficient low-rank compression and transfer learning models for geospatial applications.

## Publications 

- [Dynamical Low-Rank Compression of Neural
Networks with Robustness under Adversarial Attacks](https://arxiv.org/pdf/2505.08022)
- [Dynamic Low-Rank Training with Spectral Regularization: Achieving
Robustness in Compressed Representations](https://openreview.net/pdf?id=yZY0w0Nr7E)


## Installation and package management

1. Clone the  Github repository: 
    ```
    git clone https://github.com/ScSteffen/RobustDLRT.git
    ```

2. Create a local python environment and install the python requirements in a local virtual environment:

    ```
    python3 -m venv ./venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Example run
```
sh run_example.sh
```


## Testing Pipeline for OReole-FM
To run experiments on OReole-FM MR models, the transformers need to be converted from the timm format to transformers format. 

Run `convert_timm_to_hf_vit.py` with the appropriate command line arguments.

## Authors  (alphabetically ordered)

Main contributors
- Schnake, Stefan
- Schotthoefer, Steffen
- Yang, Lexie H. 

Student contributors
- Snyder, Thomas
- Park, Hannah
    
