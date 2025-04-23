This branch modified the model structure and used a CNN and MLP fusion model, where CNN learns image information and MLP learns hand point information extracted by Mediapipe. The CNN training set remains unchanged, and the MLP training set uses the points extracted from the original dataset through 123D_extract_keypoints.py.  

The new training set used color jepg images instead of grayscale images:  
https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data  

The complete training set and test set contain the keypoints links are as follows. How to use: Put them in the same directory as the model  
https://drive.google.com/file/d/1NoU9AfUhlmbvTh8knvmPwnEs_mdmw_Fp/view?usp=sharing  

The trained model file link is as follows:
https://drive.google.com/file/d/1OkuzSVnq1qsfPyEQGJUnsTGXNUkfFimv/view?usp=sharing  

Make sure you have the following libraries installed:  
pip install numpy pandas opencv-python mediapipe tensorflow psutil  

How to train the model:  
python Cnn_Mlp_Model.py   
  
How to load trained model(download from https://drive.google.com/file/d/1OkuzSVnq1qsfPyEQGJUnsTGXNUkfFimv/view?usp=sharing) and call the camera to participate in real-time recognition:  
python realtime_asl.py  
