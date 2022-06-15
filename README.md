Overview
This repostitory holds a code base for the research work a dynamic video saliency identification, which is still a work in progress. It is mainly inspired by the attentive CNN-LSTM based architecture proposed in [1]. We are improving its performance by tweaking its preprocessing layer/s and the overall architecture. 

Imprint

Paper: Interactive Audio-Visual Saliency Identification
Dataset: DHF1K
State of the Art: ACL




Dependencies
# Name                    Version                   Build  Channel
keras                     2.4.3                hd3eb1b0_0  
keras-preprocessing       1.1.2              pyhd3eb1b0_0  
matplotlib                3.5.1            py39h06a4308_1  
numpy                     1.22.3           py39he7a7128_0  
numpy-base                1.22.3           py39hf524024_0  
opencv-python             4.5.5.64                 pypi_0    pypi
pillow                    9.1.1                    pypi_0    pypi
python                    3.9.12               h12debd9_1  
scipy                     1.7.3            py39hc147768_0  
tensorflow                2.4.1           gpu_py39h8236f22_0  
tensorflow-datasets       4.6.0                    pypi_0    pypi
tensorflow-estimator      2.9.0                    pypi_0    pypi
tensorflow-examples       40aab88c8cfa5d2a945c781edf34528e8545037f-          pypi_0    pypi
tensorflow-gpu            2.4.1                h30adc30_0  
tensorflow-hub            0.12.0                   pypi_0    pypi
tornado                   6.1              py39h27cfd23_0  
yaml                      0.2.5                h7b6447c_0  



Folder structure
repo --> audiovisual_saliency
Video Folders
Training (DHF1K 001 - 600 videos)  Location: repo   ./DHF1K/training/
Validation (DHF1K 601 - 700 videos) Location: repo ./DHF1K/validation/
Test       (DHF1K 701- 1000 videos) Location:repo ./DHF1K/test/


Procedures
First run the extract_frames.py python file to extract video frames. This file puts frames in their corresponding video name directory as shown above
To train
In the main.py make sure the phase_gen variable is set to 'train'.
Check if the file paths after the completion of extract_frames.py is correct
run main.py

To test
In the main.py file, change the phase_gen to 'test'
run the main.py file

Outputs
For the training phase,  it saves epoch level chekpoints in the same folder as main.py.
For the testing phase, it generates prediction maps to corresponding test folder under the name 'saliency'


Task in progress
We are trying to enhance the model performance by using some preprocessing and parameter tuning. 








