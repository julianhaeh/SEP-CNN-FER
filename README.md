# SEP-CNN-FER

In this project we train a Convolutional Neural Network on a FER task from scratch. In order to accomplish this we explore different optimizations of the training pipeline and evaluate a variety of architectures. The final model is further analyzed using methods of Explainable AI, and presented in multiple demo scripts including a webcam demo.

Once the project is setup, the webcam demo can be tested by running: "Python -m Demo.infer_webcam".
There is also an video demo: "Python -m Demo.inference_video.py --input test_video.mp4 --output result_with_emotions.mp4", where test_video.mp4 is the path for the input video and result_with_emotions.mp4 is the name of the output file,
and a CSV demo "Python -m Demo.infer_csv --input_dir csvtestdir --output_csv OutputCSV.csv", where csvtest is the input directory with images and OutputCSV.csv is the CSV the output is stored in. Input files are read from root directory and output files are written to the root directory.



How to Setup: 


    1. Clone the repository

        git clone https://github.com/julianhaeh/SEP-CNN-FER

    1. Setup a new virtual enviorment: 

        conda create -n "SEP-CNN-FER" python=3.12.0
        conda activate "SEP-CNN-FER"

        Make sure the 3.12.0 (SEP-CNN-FER) Python interpreter is selected

    2. Install the dependencies:

        We will use pip to install the dependencies:
        conda install pip

        Install your pytorch version (See https://pytorch.org):
        Example: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

        Install the rest of the dependencies:
        pip install grad-cam
        pip install optuna
        pip install ultralytics
        pip install datasets

    3. Install the project in editable mode (This allows us to import the files as all other pip modules):

        Ensure your terminal is in the project's root directory (SEP-CNN-FER). If necessary, navigate there:
        Example on windows: cd SEP-CNN-FER

        pip install -e .


    No Data needs to be downloaded in advance, as it is loaded from HuggingFace. 
    Always make sure that the (SEP-CNN-FER) Python interpreter is selected and to open 
    the repository from the project's root directory (SEP-CNN-FER), torch.save or 
    torch.load may cause trouble otherwise.

    
