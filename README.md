# Application-and-practice-in-neural-networks-

## Project title 
Sleep stage classification based on EEG.

### Project introduction
Electroencephalography(EEG) is device measured brain signal. Recently machine larning used in exploting complex brain signal. 
Sleep stage classification is one of the applications, it's a challenging problem.

### Dataset description
SleepEDF 20
There are more detailed imformation. 
https://www.physionet.org/content/sleep-edfx/1.0.0/

We only use 20 patient's data. Data consist of bio signals meaured 1 ~ 2 night. Similar to previous work, we use only two EEG signal and split signal to 30 seconds.
Detailed preprocessing codes is in below link.
https://github.com/ycq091044/ManyDG/blob/main/data/sleep/sleep_edf_process.py

Many previous work experiment leave one out cross validataion bacause highly patient variation (i.e, repeat 20 times using 19 patients as train data and the remaining one as test data).
It's hard to share due to data capacity issues, so we use only five patient data. We 1 ~ 4 patient data split into train and validation set in the 8:2 ratio and 5 patient used test dataset.


+ Summary
  + data shape is (2,3000) (2EEG, (30s* 100Hz))  
  + Label : Wake, N1, N2, N3, REM
  + Train: 80% of patient 1 ~ 4
  + Validataion: 20% of patient 1~4 
  + Test: patient 5
  + Train,validation is npz files ('x':data, 'y' is label)
  + Test is npy files

