# Chinese Sentence Completion
The datasets and trained toy model can be found at : https://drive.google.com/drive/folders/1FJec5tUuV9XmIokmN45hEIQL9XB15QSG?usp=drive_link

## To try out the models, follow these steps:

1. Git clone the repository to your local machine.

2. Download a model from the provided link and place the associated files in the same directory as the repository.

3. Inside the tryout.py file, update the model hyperparameters to match the ones in the config file of the model.

4. Scroll down and modify the beginning of the sentence to the desired completion.

## To train your own model, follow these steps:

1. In the code file multihead_attention_text_generation.py, adjust the model hyperparameters as desired. If it is your first time running, you need to set the variable FIRST_TIME_RUNNING = True and NEED_TOKEN_CALCULATION = True.

2. Place the training data text file in the same directory and start running the code. If training gets interrupted and you want to continue from where you left off, set the variable FIRST_TIME_RUNNING = False and NEED_TOKEN_CALCULATION = False to resume training.

3. During training, the model and auxiliary code will be saved for future use.

We provided the toy training dataset  xiaoshuo.txt to start training quickly.
