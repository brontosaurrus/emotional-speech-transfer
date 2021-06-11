# emotional-speech-transfer

This repository is using the EMOV-DB dataset https://mega.nz/folder/KBp32apT#gLIgyWf9iQ-yqnWFUFuUHg. All wav files should sit under the database/emoSpeech/all directory. Other datasets can be used with new preprocessing scripts and --dataset tag in step 3. The metadata.csv file should also be in the databse/emoSpeech directory. 

### 1. Install Dependancies

Install Python3, Tensorflow 1.15 and CUDA 10.


### 2. Install Requirements
```
pip install -r requirements.txt

```
### 3. Preprocess Data
```
python3 preprocess.py --dataset emospeech
```

### 3. Training
```
python3 train.py
```


### 4. Inference from a Checkpoint

Through training checkpoint files can be found under the logs-tacotron directory. Any checkpoint can be tested in this directory with the following structure. The step count needs to be changed (eg. logs-tacotron/model.ckpt-185000 ). Any text can be inserted after the '--text' tag, however longer or multiple sentences are less effective. '--reference_audio' represents the directory to a .wav audio file with the voice/emotion that would be preferred to transfer onto. 
```
python3 eval.py --checkpoint logs-tacotron/model.ckpt-XXXXX --text "hello text" --reference_audio /path/to/ref_audio
```
