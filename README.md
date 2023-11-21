# mc-lstm-time-series
本项目是论文《[Anomaly Detection Using Multiscale C-LSTM for Univariate Time-Series](https://www.hindawi.com/journals/scn/2023/6597623/)》的实验代码，实现了多种时间序列异常检测模型。<br>
This project is the experimental code of the paper "*[Anomaly Detection Using Multiscale C-LSTM for Univariate Time-Series](https://www.hindawi.com/journals/scn/2023/6597623/)*", which implements a variety of time series anomaly detection models.

## 目录 Table of Contents

- [项目目录 Project Directory](#项目目录-project-directory)
- [使用方法 Getting Started](#使用方法-getting-started)
- [项目声明 Project Statement](#项目声明-project-statement)

<h2 id="project">项目目录 Project Directory</h2>

├─ datasets (数据集目录 Dataset directory)<br>
&emsp;├─ Numenta Anomaly Benchmark (NAB数据集目录 NAB dataset directory)<br>
&emsp;├─ Yahoo! Webscope S5 (雅虎数据集目录 Yahoo dataset directory)<br>
├─ dl_models (模型目录 Model directory) <br>
&emsp;├─ cnn.py (CNN模型 CNN model)<br>
&emsp;├─ ms_cnn.py (多尺度CNN模型 Multi-scale CNN model)<br>
&emsp;├─ c_lstm.py (C-LSTM模型 C-LSTM model)<br>
&emsp;├─ c_lstm_ae.py (C-LSTM-AE模型 C-LSTM-AE model)<br>
&emsp;├─ imc_lstm.py (IMC-LSTM模型 IMC-LSTM model)<br>
&emsp;├─ cmc_lstm.py (CMC-LSTM模型 CMC-LSTM model)<br>
&emsp;├─ smc_lstm.py (SMC-LSTM模型 SMC-LSTM model)<br>
├─ dataPreprocessing.py (数据预处理 Data preprocessing)<br>
├─ train.py (训练代码 Training data)<br>
├─ train_windows.py (不同滑动窗口大小的训练代码 Training code for different sliding window sizes)<br>
├─ requirements.txt (项目依赖 Project dependencies)<br>

> 以上列出了模型文件及主要的训练代码文件，其余未列出的文件均为项目基础文件，无需重点关注。<br>
> The model files and main training code files are listed above, and the rest of the unlisted files are the basic files of the project and do not need to be paid attention to.<br>
> 本项目使用的数据集是网上公开的数据集，并非私有。因此，为了维护数据集的版权，我们并未将数据集一并上传。数据集的原链接如下：<br>
> The datasets used in this project are publicly available online, not private. Therefore, in order to maintain the copyright of the dataset, we did not upload the dataset together. The original link to the dataset is as follows:<br>
> NAB: https://github.com/numenta/NAB<br>
> Yahoo: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70

<h2 id="get-start">使用方法 Getting Started</h2>

首先，拉取本项目到本地。<br>
First, pull the project to the local.

    $ git clone git@github.com:lyx199504/mc-lstm-time-series.git

接着，进入到项目中并安装本项目的依赖。但要注意，pytorch可能需要采取其他方式安装，安装完毕pytorch后可直接用如下代码安装其他依赖。<br>
Next, enter the project and install the dependencies of the project. However, it should be noted that pytorch may need to be installed in other ways. After installing pytorch, you can directly install other dependencies with the following code.

    $ cd mc-lstm-time-series/
    $ pip install -r requirements.txt

然后，分别将NAB和雅虎数据集下载到项目的NAB数据集目录和雅虎数据集目录中。<br>
Then, download the NAB and Yahoo datasets to the project's NAB dataset directory and Yahoo dataset directory, respectively.

最后，执行train.py或train_windows.py即可训练模型。<br>
Finally, execute train.py or train_windows.py to train the model.

<h2 id="statement">项目声明 Project Statement</h2>

本项目的作者及单位：<br>
The author and affiliation of this project:

    项目名称（Project Name）：mc-lstm-time-series
    项目作者（Author）：Yixiang Lu, Yudan Cheng, Jianbin Mai, Hongliang Sun, Juli Yin, Guoxuan Zhong
    作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security, Jinan University）

本实验代码基于param-opt训练工具，原项目作者及出处如下：<br>
The experimental code is based on the param-opt training tool. The author and source of the original project are as follows:<br>
**Author: Yixiang Lu**<br>
**Project: [param-opt](https://github.com/lyx199504/param-opt)**

若要引用本论文，可按照如下latex引用格式：<br>
If you want to cite this paper, you could use the following latex citation format:

    @article{lu2023anomaly,
        title={Anomaly Detection Using Multiscale C-LSTM for Univariate Time-Series},
        author={Lu, Yi-Xiang and Jin, Xiao-Bo and Liu, Dong-Jie and Zhang, Xin-Chang and Geng, Guang-Gang and others},
        journal={Security and Communication Networks},
        volume={2023},
        year={2023},
        publisher={Hindawi}
    }

