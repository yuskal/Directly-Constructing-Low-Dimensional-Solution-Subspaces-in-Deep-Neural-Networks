## Getting Started

Follow these steps to set up the environment and run the experiments.

### 1. Installation

First, install the required dependencies using `pip`. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

All main scripts are located in the scripts/ directory. Each script is self-contained and handles data loading, preprocessing, and the training loop.
Navigate to the scripts folder:
```bash
cd scripts
```

Now you can run the experiments directly:

For CIFAR-100
```bash
python c100_main.py
```

For ImageNet
```bash
python imagenet_main.py
```

For MNLI
```bash
python mnli_main.py
```

3. Custom Seeds (Optional)

For reproducibility or testing across different initializations, you can manually set a seed using the --seed flag:
# Example: Running MNLI with a specific seed
```bash
python mnli_main.py --seed 42
```

Note: If no seed is provided, the scripts will use the default configuration defined within the files.

