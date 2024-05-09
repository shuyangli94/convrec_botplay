To set up your environment:

```bash
export HOME_DIR=YOUR/HOME/DIR

# Set up venv
cd ${HOME_DIR}
python3 -m venv ${HOME_DIR}cr-env

# Source
source ${HOME_DIR}cr-env/bin/activate

# Basic requirements
python3 -m pip install --upgrade pip
pip3 install --upgrade wheel 
pip3 install Cython==0.29.26
pip3 install numpy==1.21.5 scipy==1.7.3
pip3 install nvidia-ml-py3

# cuda 10.2, torch version 1.10
pip3 install torch==1.10.1

# cuda 11.3, torch version 1.11
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install --upgrade tqdm jupyterlab
pip3 install --upgrade nltk
python3 -c "import nltk;nltk.download('punkt');nltk.download('popular')"
pip3 install transformers==4.15.0 sentencepiece==0.1.96
pip3 install tensorboard

pip3 install scikit-learn==0.24.2
pip3 install --upgrade yake ujson PyYAML

# Assorted
pip3 install --upgrade pandas omegaconf
# pip3 install pytorch-lightning==0.7.1
pip3 install pytorch-lightning==1.5.10
pip3 install tensorflow==1.12.0 tensorflow-gpu==1.12.0
pip3 install fbpca
pip3 install --upgrade hydra-core
```