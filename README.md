# PD Classification
In this project,  we use a two-view model with feature aggregation to classify PD patients using speech data. 

Two experiments are conducted here. In ***pddiagnose***, we try to classify PD patients with healthy people, and in ***pdseverity***, we try to classify disease severity of PD patients.

### Data

[MDVR-KCL](https://zenodo.org/record/2867216#.YW57GNy-uUl)

- white space and voice of the staff are cropped in our ***KCL dataset optimized***

### Model

In ***model***, *one-dim/two-dim/dual-view* CNN models with feature aggregation are used in our project to classify PD patients. 

### Low-level data initial

- `diagnose_waveform_initial.py` and `diagnose_mel_spec_initial.py` in ***pddiagnose*** are used to initial waveform segments and Mel-spectrograms of PD patients(PD) and healthy controllers(HC). 
- `severity_waveform_initial.py` and `severity_melspectrogram_initial.py` in ***pdseverity*** are used to initial waveform segments and Mel-spectrograms of PD patients(PD) only.

### Run our model

For example, in ***pddiagnose*** : 

- In `test_onedcnn.py` , the training data and testing data should be placed in:

  ```python
  def getaddr(x):  #line 22
      return "F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\train",\
             "F:\Parkinson speech\dataset\\five fold cross validation\\" + str(x) + "\\test"
  ```

  Or you can change the loading address of training and testing in :  

  ```python
  train_addr, test_addr = getaddr(fold)   #line 39
  ```

  Different models are used in : 

  ```python
  model = onedcnn().to(device)    #line 65
  #model = aggregation_onedcnn().to(device)
  ```

### Environment

Python  3.8   pytorch  1.7.0   torchvision  0.8.1   

numpy  1.19.2   librosa  0.8.1   prettytable  2.1.0

### Update

2021/10/19

