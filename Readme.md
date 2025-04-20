The CNN model and MLP model in the paper and the training and test sets used to train and test the models are recorded here.  
Reference training set data link now:  
https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download  

The project plans to use a new training set, using color jepg images instead of grayscale images:  
https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data  

The complete training set and test set links are as follows. How to use: Unzip and put them in the same directory as the CNN model  
https://drive.google.com/file/d/1QkCedJrVkm0wkH7O3nrmEnQOJwXYeHCY/view?usp=drive_link  

Make sure you have the following libraries installed:  
pip install numpy pandas opencv-python mediapipe tensorflow psutil  

How to use:  
python CNN_Model.py  
python evaluate_trained_model.py  
python realtime_asl.py  


