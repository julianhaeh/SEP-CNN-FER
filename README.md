# SEP-CNN-FER

In this project we train a Convolutional Neural Network on a FER task from scratch. In order to accomplish this we explore different optimizations of the training pipeline and evaluate a variety of architectures. The final model is further analyzed using methods of Explainable AI, and presented in multiple demo scripts including a webcam demo.


How to Setup: 


    1. Clone the repository

        git clone https://github.com/julianhaeh/SEP-CNN-FER

    1. Setup a new virtual enviorment: 

        conda create -n "SEP-CNN-FER" python=3.12.0
        conda activate "SEP-CNN-FER"

    2. Install the dependencies:

        We will use pip to install the dependencies:
        conda install pip

        Install your pytorch version (See https://pytorch.org):
        Example: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

        Install the rest of the dependencies:
        pip install grad-cam
        pip install matplotlib
        pip install numpy ??
        pip install optuna
        pip install scikit-learn ??
        pip install ultralytics
        pip install datasets

    3. Install the project in editable mode (This allows us to import the files as all other pip modules):

        pip install -e .

    

  
        
        


    No Data needs to be downloaded in advance, as it is loaded from HuggingFace. 
