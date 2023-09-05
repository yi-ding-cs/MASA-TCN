# MASA-TCN
The PyTorch implementation of MASA-TCN

[MASA-TCN: Multi-anchor Space-aware Temporal Convolutional Neural Networks for Continuous and Discrete EEG Emotion Recognition](https://arxiv.org/abs/2308.16207)
## Example of the usage
```python
from model import MASA_TCN

data = torch.randn(1, 1, 192, 96)  # (batch_size=1, cnn_channel=1, EEG_channel*feature=32*6, data_sequence=96)

# For regression, the output is (batch_size, data_sequence, 1).
net = MASA_TCN(
        cnn1d_channels=[128, 128, 128],
        cnn1d_kernel_size=[3, 5, 15],
        cnn1d_dropout_rate=0.1,
        num_eeg_chan=32,
        freq=6,
        output_dim=1,
        early_fusion=True,
        model_type='reg')
preds = net(data)

# For classification, the output is (batch_size, num_classes). Note: output_dim should be the number of classes.
net = MASA_TCN(
        cnn1d_channels=[128, 128, 128],
        cnn1d_kernel_size=[3, 5, 15],
        cnn1d_dropout_rate=0.1,
        num_eeg_chan=32,
        freq=6,
        output_dim=2,
        early_fusion=True,
        model_type='cls')
preds = net(data)
```

# CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |

# Cite
Please cite our paper if you use our code in your own work:

```
@misc{ding2023masatcn,
      title={MASA-TCN: Multi-anchor Space-aware Temporal Convolutional Neural Networks for Continuous and Discrete EEG Emotion Recognition}, 
      author={Yi Ding and Su Zhang and Chuangao Tang and Cuntai Guan},
      year={2023},
      eprint={2308.16207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```
