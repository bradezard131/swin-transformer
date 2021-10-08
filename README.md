# Swin Transformers

Provides an implementation of Swin Transformers, attempting to adhere to best practices for reproducibility

## Creating Environment
```bash
$ conda env create -f environment.yml
$ conda activate swin
```

## Running the model
```bash
$ python scripts/train.py
```

## Importing the model into other code
```bash
pip install -e .
```

```python
from swin import build_swin_model

model = build_swin_model(out_classes=20)
```

## Running the tests
```bash
pytest
```
