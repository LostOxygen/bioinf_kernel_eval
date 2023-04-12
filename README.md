# BioInformatics Kernel Evaluation
Kernel Evaluation for Different Convolution Operations over Medical Images

## Usage:
```
python main.py [-h] [--gpu | -g DEVICE_ID] [--batch_size | -bs BATCH_SIZE] [--epochs | -e EPOCHS]
```

## Example Usage:
```python
python main.py --gpu 0 --batch_size 32 --epochs 100
```

### Training Arguments
| Argument | Type | Description |
|----------|------|-------------|
| -h, --help | None | shows argument help message |
| -g, --gpu | INT | specifies device ID to use. [0, N] for GPU and -1 for CPU (default=-1)
| -e, --epochs | INT | number of epochs to train (default=100) | 
| -bs, --batch_size | INT | batch size (default=128) |