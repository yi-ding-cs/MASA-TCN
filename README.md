# MASA-TCN
The PyTorch implementation of MASA-TCN

[MASA-TCN: Multi-anchor Space-aware Temporal Convolutional Neural Networks for Continuous and Discrete EEG Emotion Recognition](https://ieeexplore.ieee.org/document/10506986)

# Prepare the data
Download MAHNOB-HCI dataset [here](https://mahnob-db.eu/hci-tagging/). And set the data folder as the root_directory in configs.py, e.g., /home/dingyi/MAHNOB/. This folder should contains two subfolders, ./Sessions/ and ./Subjects/.

Get the continuous label in this [repo](https://github.com/soheilrayatdoost/ContinuousEmotionDetection). Put the lable_continous_Mahnob.mat at the data folder, e.g., /home/dingyi/MAHNOB/lable_continous_Mahnob.mat

Note that it might pop some error messages when you create the dataset by using generate_dataset.py. It is because there are some format errors in the original data. You can identify the file according to the error message and correct the format error in that file.

The exact folder/files to be edit include:

A. Sessions/1200/P10-Rec1-All-Data-New_Section_30.tsv - `Remove Line 3169-3176 as their format is broken`.

B. Sessions/1854 - `Remove this trial folder as it does not contain EEG recordings`.

C. Sessions/1984 - `Remove this trial folder as it does not contain EEG recordings`.

# Create the enviroment
Please create an anaconda virtual environment by:

> $ conda create --name MASA python=3.9

Activate the virtual environment by:

> $ conda activate MASA

Install the requirements by:

> $ pip3 install -r requirements.txt

Install the torch for GPU using the commend on https://pytorch.org/get-started/previous-versions/#v1110

# Run the code
Step 1: Check the config.py file first and change the parameters accordingly. Mainly, update the `"root_directory"` and `"output_root_directory"` according to your data location.

Step 2: Run generate_dataset.py.

Step 3: Check the parameters in the main.py file and change them accordingly. Mainly, update the `"-dataset_path"`, `"-load_path"`, `"-save_path"`, and `"-python_package_path"` according to your local directory.

Step 4: Run main.py to train and evaluate the network.

Step 5: Using generate_results_csv.py to get the summarized results.

Please add `pip install chardet` if you received an error saying "ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'" when running `main.py`.

## Example of the usage on other datasets
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
# Acknowledgement
Thanks to [Dr. Zhang Su](https://github.com/sucv) for his kind help with code checking and optimization. 
# CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |

# Cite
Please cite our paper if you use our code in your own work:

```
@ARTICLE{10506986,
  author={Ding, Yi and Zhang, Su and Tang, Chuangao and Guan, Cuntai},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={MASA-TCN: Multi-Anchor Space-Aware Temporal Convolutional Neural Networks for Continuous and Discrete EEG Emotion Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Electroencephalography;Kernel;Task analysis;Brain modeling;Emotion recognition;Feature extraction;Convolutional neural networks;Temporal convolutional neural networks (TCN);emotion recognition;electroencephalogram (EEG)},
  doi={10.1109/JBHI.2024.3392564}}
```
