# Application-and-practice-in-neural-networks-

## Project title 
Sleep stage classification based on EEG.

### Project introduction
Electroencephalography(EEG) is device measured brain signal. Recently machine larning used in exploting complex brain signal. 
Sleep stage classification is one of the applications, it's a challenging problem.

### Dataset description
SleepEDF 20
There are more detailed imformation. https://www.physionet.org/content/sleep-edfx/1.0.0/

We only use 20 patient's data. Data consist of bio signals meaured 1 ~ 2 night. Similar to previous work, we use only two EEG signal and split signal to 30 seconds.
Detailed preprocessing codes is in below link.
https://github.com/ycq091044/ManyDG/blob/main/data/sleep/sleep_edf_process.py
Many previous work experiment leave one out cross validataion, i.e, repeat 20 times using 19 patients as train data and the remaining one as test data.



+ Summary
  ++ data shape is (2,3000) (2EEG, (30s* 100Hz))  
  ++ 


  + dddd
