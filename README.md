Legend:
DIR = means this is a directory
\- subullets are descriptions
\* points out important files

Directories
- DIR processing: stores all the image processing data
  - DIR .ipynb_checkpoints: jupyterlab makes these, just ignore them
  - process_csv.py
    - processes the labels for the training data
  - process_images.py
    - processes the train images, crops them
  - process_test.ipynb
    - processes the test images, crops them (same as process_images but for test data)
  - process_images_notebook.ipynb
    - takes the npy arrays of 5,000 images and splits it to have individual images
    - hopefully won't need to runt this again
  - process_labels.py
    - open processed csv and peals out labels to put it into its own npy file
  - train.csv
    - train labels before processing
  - train_processed.csv
    - train labels after processing
  - *data_to_npy.sbatch
    - Example sbatch if you want to run something with SLURM on the HPC
- DIR models: store all the models
  - CheXpertDataset.py 
    - Dataset class for our data
  - initial_model.ipynb
    - Our first CNN
- DIR run_test: run the model on the test data so that we can submit it to the class leaderboard
  - run_test_data.ipynb
    - takes the test data and runs whatever model you give (loads the model weights)
    - output: CSV file that can be submitted for the leader board
- .gitignore
  - helps us ignore random checkpoints made by jupyter notebooks






