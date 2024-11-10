INSTRUCTIONS TO RUN RESNET 34.

1) To run the algorithm for the classification of cats vs dogs, the first 500 images from each of the classes are selected for data augmentation using the Augmentation Code.py. 

2) Afterwards, 500 augmented images each belonging to cats and dogs are manually transferred into their respective class folder under the main folder named train_2000 which already has 1500 raw images for each of the 2 classes for further data preprocessing.

3) Next, the sequence for runing the AI Project_Resnet34.py is split into 2 portions. Firstly, besides importing the necessary libraries and loading up of the data, run the data preprocessing of train and validation images portion up till the portion that prints out the size of the training and validation data being used.
"
print(f'Training images count: {dataset_sizes["train"]}')
print(f'Validation images count: {dataset_sizes["val"]}')
print(f'Class names:{class_names}')"

4) To start training the model, run the remaining part of the code which will define the model being used which is resnet34 and the desired amount of epochs and learning rate, the model will then make use of the weights of the epoch that has achieved the best validation accuracy and the least validation loss for the testing stage.
   
5) Included in the code, are functions that plots randomly selected images and dsiplay how the model classify them as, and a function that allows you to individually check how the model classify the desired image to be either a dog or a cat.
