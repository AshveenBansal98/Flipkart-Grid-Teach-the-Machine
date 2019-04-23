This notebook was made on google colab.Make sure to change runtime to GPU. 
As the memory limit is 12GB, running all cells together on colab will crash runtime. So the notebook should be partially run and the session needs to be reset
The cells corresponding to taking the input need to be run everytime the session is started

test and training images were split into two zips
You need to make a copy of the files in your drive and then replace the link with corresponding links of your id

The images were scaled to 48*64 size
Input was normalised by subtracting and dividing by 128.0

The model was tested with 1, 2 and 3 CNN layers and 2 layers were choosen
Then different kernel size were tested and 72 for first and 88 for second layer was choosen
Then different dense layer size were tested and 2048 was choosen
Dropout was tested but it didn't improve result
Converting second CNN layer (88C5) to two CNN layers (88C3-88C3) didn't improved result
2048 dense layer was split into two, 1024 and 128 gave best result 

The initial cells correspond to above testing and are marked with "MODEL TESTING"

The model was run for 2000 epochs in two phases
After training was completed, model with lowest val_loss was selected for prediction
The final prediction values were rounded off to nearest integer

TO TEST ONLY FINAL MODEL, SKIP THE CELLS MARKED "#MODEL TESTING"

As weights are randomly assigned, it should be run for 2-3 time for best results
Final model is 72C5-MaxPool2d-88C5-MaxPool2d-2048-128-4

model = Sequential()
model.add(Conv2D(72,kernel_size=5,padding = 'same', activation='relu', input_shape=(image_height, image_width, channels)))
model.add(MaxPool2D())
model.add(Conv2D(88,kernel_size=5,padding = 'same',activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='relu'))
model.compile(optimizer="adam", loss="mean_squared_error")

Link to final model weights file: https://drive.google.com/open?id=1ZRcnkQMU2Td9BSEijmD632bwtN28KlLR

TO TEST ONLY FINAL MODEL, SKIP THE CELLS MARKED "#MODEL TESTING"