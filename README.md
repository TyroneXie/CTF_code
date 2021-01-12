# CTF



### CTF: Anomaly Detection in High-Dimensional Time Series with Coarse-to-Fine Model Transfer

CTF is a coarse-to-fine model transfer based framework to achieve a scalable and accurate data-center-scale anomaly detection. 
It pre-trains a coarse-grained model, uses the model to extract and compress per-machine features to a distribution, clusters machines according to the distribution, and conducts model transfer to fine-tune per-cluster models for high accuracy.  



## Getting Started

#### Clone the repo

```
git clone https://github.com/smallcowbaby/CTF_code
```

#### Get data


You can get the public dataset (CTF data) using:

```shell
git clone https://github.com/smallcowbaby/CTF_data && cd CTF_data && cat CTF_data.tar.gz.* | tar -zxv
```

#### Install dependencies (with python 3.6) 

(virtualenv is recommended)

```shell
pip install -r requirements.txt
```

#### Run the code

Put the folders `CTF_data`, `label_result` and `CTF_code` in the same folder. Run the following code:  

```
cd CTF_code && python run_main_transfer.py && python download_score.py && python get_POT_param.py
```

If you want to change the default configuration, you can edit `ExpConfig_transfer` in `run_main_transfer.py`.


## Processing

With the default configuration, `run_main_transfer.py` follows these steps:

* Train CTF models with training set, and validate at a fixed frequency. Early stop method is applied by default.
* Test the model on the testing set, and save anomaly score in `../result/result_for_period2_*`.
* Get the scores from each cluster and save them in `label_data`.
* Run POT models in each cluster to find the threshold of anomaly score, and using this threshold to predict on the testing set.


