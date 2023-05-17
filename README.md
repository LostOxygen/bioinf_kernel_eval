# BioInformatics Kernel Evaluation
Performance Evaluation between Normal and Depthwise Seperable Convolutions for Medical Image Classification.

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
| -lr, --learning_rate | FLOAT | learning rate (default=0.1) |
| -m, --model_type | STRING | model type to use [vgg11, vgg13, vgg16, vgg19, resnet34] (default='vgg11') |
| -d, --depthwise | BOOL | use depthwise separable convolutions (default=False) |
| -ev, --eval_only | BOOL | evaluate model only (default=False) |
| -no, --normalize | BOOL | normalization (default=False) |
