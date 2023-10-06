import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tqdm import tqdm
import os
import logging
import gc
import time
from datetime import datetime
from load_image_data import ImageDataset 
  
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(2)

"""
Dataset source:
    https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy
"""


"""
WGANs-GP to increase sizes of small samples for mild impairement cases
"""

def show_pictures_of_data(x_train, y_train, width, height):
    # previous_text = None
    # for i,x in enumerate(x_train):
    #     if previous_text: previous_text.set_visible(False)
    #     previous_text = plt.text(0,-10,s=classes[y_train[i]])
    #     plt.imshow(x.reshape(width,height,3))
    #     plt.show()
    pass

def reduce_size_of_dataset(N, x_train, y_train, x_test, y_test, x_cv, y_cv):
    print(f"shuffling training data and indices")
    train_indices = np.arange(int(N*x_train.shape[0]))
    np.random.shuffle(train_indices)
    
    print(f"shuffling testing data and indices")
    test_indices = np.arange(int(N*x_test.shape[0]))
    np.random.shuffle(test_indices)
    
    pre_reduce_samples = x_train.shape[0]
    x_train, y_train = x_train[train_indices], y_train[train_indices]
    x_test, y_test, x_cv, y_cv = x_test[test_indices], y_test[test_indices], x_cv[test_indices], y_cv[test_indices] 
    post_reduce_samples = x_train.shape[0]
    
    if pre_reduce_samples <= post_reduce_samples and post_reduce_samples + len(pre_reduce_samples)*N == pre_reduce_samples:
        assert f"Unable to reduce samples by factor {N*100}%"
    
    return x_train, y_train, x_test, y_test, x_cv, y_cv
    

def load_data(N):
    path = 'data/'

    train = ImageDataset(PATH=f'{path}/train', TRAIN=True)
    test  = ImageDataset(PATH=f'{path}/test', TRAIN=False)
    
    # Load dataset
    x_train, y_train = train.load_data()
    x_test, y_test = test.load_data()
    x_cv, x_test = np.array_split(x_test, 2)
    y_cv, y_test = np.array_split(y_test, 2) 

    x_train, y_train, x_test, y_test, x_cv, y_cv = reduce_size_of_dataset(N, x_train, y_train, x_test, y_test, x_cv, y_cv)
    
    num_classes = train.get_num_categories()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test  = tf.keras.utils.to_categorical(y_test, num_classes)
    y_cv    = tf.keras.utils.to_categorical(y_cv, num_classes)
     
    width, height = train.get_image_dimensions()
    show_pictures_of_data(x_train, y_train, width, height)
    
    return num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv
    
class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') >= 0.995:
            self.model.stop_training = True
    
def create_model(num_classes):
    tf.random.set_seed(1234) 

    # Create tensorflow model
    model = Sequential(
        [
            # Convolution and Pooling 3 times before flattening to reduce total number of pixels passed to last Dense layer
            Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),  # relu - ReLU(x)=max(0,x)
            MaxPooling2D(2,2),
            # Dropout(0.15), 
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(2,2), 
            # Dropout(0.25),                                                  # select a percentage of outputs at random and set to 0, helps regularize
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),                                                      # take 4D array, turn into vector
            Dense(128, activation='relu'),
            # Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ]
    )
    
    print(model.summary())

    return model
    
    
def main():
    os.chdir("/home/jtrull/python/alzheimers")
    
    f = open('histories.txt', 'a')
    f.write(f"\ntest run: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
    f.write(f"accuracy percent,percent of data,batch size,learning_rate,history.params,history.history['acc'],history.history['loss'],elapsed_time\n")
                    
    f.close()
    
    # plot features
    #   no plotting features needed - dataset consists of images
    #       features are just intensity values of pixels

    data_subets =   [ 0.5, 0.75, 0.99 ]                                             # may need to downsample images before using more data
    epochs =        [ 10, 25, 50, 75, 100 ]                                      # 'random' scaling numbers
    batch_sizes =   [ 32, 64, 128 ]                                          # powers of 2
    learning_rate = 0.001
    for d in data_subets:
        for e in epochs:
            for b in batch_sizes:
                model = None
                try:
                    start = time.time()
                    num_classes, x_train, y_train, x_test, y_test, x_cv, y_cv = load_data(d)
                    model = create_model(num_classes)
                    
                    # Compile model
                    model.compile(
                        loss=tf.keras.losses.categorical_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        metrics=['acc']
                    )
                    
                    model.build()
                    print(f"Model params:{model.count_params()}")

                    callbacks = callback()
                    history = model.fit(x_train, y_train,
                            batch_size=b,
                            epochs=e,
                            verbose=1,                                  # type: ignore
                            validation_data=(x_cv, y_cv),
                            callbacks=[callbacks])
                    loss, acc = model.evaluate(x_test, y_test, verbose=0)   # type: ignore
                    
                    print('Test loss:', loss)
                    print('Test accuracy:', acc)
                    end = time.time()
                    elapsed_time = f'{(end-start):.0f}'
                    acc_perc = f'{int(acc*100)}%'
                    data_perc = f'{int(d*100)}%'
                    f = open('histories.txt', 'a')
                    
                    if acc >= 0.95:
                        model.save(f'models/alz_cnn_{e}_es_{b}_bs_{learning_rate}_lr_{data_perc}_data_{acc_perc}_acc_{loss:.2f}_loss_{elapsed_time}_seconds.keras')
                        f.write(f"* {acc_perc}, {data_perc}, {b}, {history.params}, {history.history['acc']}, {history.history['loss']}, {elapsed_time}\n")
                        continue
                    elif acc >= 0.5:
                        f.write(f"{acc_perc}, {data_perc}, {b}, {learning_rate}, {history.params}, {history.history['acc']}, {history.history['loss']}, {elapsed_time}\n")
                    else:
                        print(f"skipping write of model with acc:{acc_perc}")
                        
                    f.close() 
                    print(f"skipping model with accuracy: {acc_perc}")
                    
                    del history 
                except Exception as ex:
                    # write out failure
                    f = open('failures.txt', 'a')
                    f.write(f"{(d,b,e)}, {ex}\n")
                    f.close()
                    time.sleep(5)
                finally:
                    # garbage collection
                    if model:
                        del model   
                    K.clear_session()
                    gc.collect() 

if __name__ == '__main__': 
    main()