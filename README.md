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

## For Windows demo specifically

1. Open the `run` folder in File Explorer:

    - Double-click `webcam.bat` to start the webcam demo.
    - Drag and drop an `.mp4` file onto `video.bat`.
    - Drag and drop an image folder onto `csv.bat`.

2. Outputs (videos and CSV files) will be saved in the `outputs` folder.

## Demo commands cross platform (run from repo root)

### macOS / Linux
    
If python doesn’t work, try python3

    Webcam:
    python3 -m Demo.infer_webcam --weights "./Experiments/Models/ReducedClassifier_Weighted_CE_EntireData.pth" --flip

    Video:
    python3 -m Demo.infer_video --input "/path/to/input.mp4" --output "/path/to/out.mp4" --weights "./Experiments/Models/ReducedClassifier_Weighted_CE_EntireData.pth"

    CSV:
    python3 -m Demo.infer_csv --input_dir "/path/to/images_folder" --output_csv "/path/to/preds.csv" --weights "./Experiments/Models/ReducedClassifier_Weighted_CE_EntireData.pth"

### Windows

    Webcam:
    python -m Demo.infer_webcam --flip

    Video:
    python -m Demo.infer_video --input "C:\path\to\input.mp4" --output "C:\path\to\out.mp4"
    
    CSV:
    python -m Demo.infer_csv --input_dir "C:\path\to\images_folder" --output_csv "C:\path\to\preds.csv"