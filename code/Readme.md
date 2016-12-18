The mv-lstm is implemented by modifying the gLSTM code [1]. The gLSTM code uses Neuraltalk [2] implementation.  The gLSTM is an image caption generation implementation using the LSTM. This is modified for the classification problem. 

data/mmdb_book:

1. The mmdb_book_fps_30_samplesize_25.mat is the dataset containing HOG, HOF and Headpose features. The features are extracted from 59 video sessions of adult-child interactions. Please contact author to obtain the dataset. 

2. The json files contain train and test split information. 59 files are present corresponding to 59-fold validation. The original neuraltalk codebase needs the train and test split information is a particular format. Therefore, these json files are created. In future, this needs to be changed for better management.
 
train.py: Train a model using the input features in the data/mmdb_book folder.

test.py: Test using split information in the json  files.

cv: The train.py stores the model file here.

status: test.py stores the testing output here.

imagernn/mvlstm.py: The core mvlstm implementation.

imagernn/mvlstm_batch.py: Forms a batch and invokes the mvlstm functions.

imagernn/dataprovider.py: Dataset load management.

solver.py: optimization methods implementation.

*utils.py: Supporting functions.


To train a model
----------------
python train.py

To test a model
---------------
python test.py





[1]
Xu Jia, Efstratios Gavves, Basura Fernando, Peter Young, Tinne Tuytelaars. "Guiding Long Short-Memory for Image Caption Generation".ICCV 2015.
* It is built on Andrej Karpathy's Neuraltalk implementation [1].


[2] Neuraltalk: https://github.com/karpathy/neuraltalk

