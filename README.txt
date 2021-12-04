DL platform: PyTorch
device: CPU

code files: preprocess.py, BiLSTM_CRF.py, training.py, test.py
model files: cws.model, cws_all.model
result file: res.txt

How to run: Put code files and model files in the same directory.
  Cause training and testing are separated, you can train the model by running training.py,
  and then load the model and do testing by running test.py. Results will be written into res.txt.