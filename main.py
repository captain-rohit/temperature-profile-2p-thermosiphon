import pandas as pd
import numpy as np
import os
import time
import fire
import sys
sys.setrecursionlimit(150000)


from keras.models import Sequential, load_model

from keras.layers import Dense
from tensorflow.keras import initializers, activations, optimizers
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

 

DATASET_PATH = '/home/rohit/Projects/temp-profile-thermosiphon/dataset.csv'
DIR = '/home/rohit/Projects/temp-profile-thermosiphon'
PredictorScaler = None
TargetScaler = None



def preprocess_data():
    temperatures = pd.read_csv(DATASET_PATH)

    # print(temperatures.head())
    predictors = ['Heat Flux', 'Submergence', 'Distance' ]
    target = ['Temperature']
    x_data = temperatures[predictors].values
    y_data = temperatures[target].values
    global PredictorScaler, TargetScaler
    PredictorScaler=StandardScaler().fit(x_data)
    TargetScaler=StandardScaler().fit(y_data)

    x_data = PredictorScaler.transform(x_data)
    y_data = TargetScaler.transform(y_data)

    return x_data, y_data

def load_model(path: str):
    path = os.path.join(DIR, path)
    if not os.path.exists(path):
        print("!!! Given model does not exist !!!")
        return
    model = load_model(path)
    print('hiiiii')
    model.summary()
    print('   ')
    print('   --------------------------------    ')
    print('    ')
    evaluate_model(model)
    return

def evaluate_model(model):
    
    x_data, y_data = preprocess_data()
    _, X, _, Y = train_test_split(x_data, y_data, test_size=0.5, random_state=42)
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

def train_model():

    x_data, y_data = preprocess_data()

    # print(y_data)
    # separated_x = {}
    # separated_y = {}

    # for i in range(len(x_data)):
    #     x = x_data[i]
    #     y = y_data[i]
    #     if not x[1] in separated_x:
    #         separated_x[x[1]] = []
    #     separated_x[x[1]].append(x)
    #     if not x[1] in separated_y:
    #         separated_y[x[1]] = []
    #     separated_y[x[1]].append(y)

    # for sub in separated_x.keys():
        # x = np.array(separated_x[sub])
        # y = np.array(separated_y[sub])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)
    ann_model = Sequential()
    print(x_train.shape)
    ann_model.add(Input(shape=x_train.shape))

    ann_model.add(Dense( units = 10,
                    input_dim = x_data.shape[1],
                    input_shape = x_train.shape,
                    kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=0.02), 
                    bias_initializer = initializers.Zeros(),
                    activation = activations.tanh
                        ))
    ann_model.add(Dense( units = 10, 
                    kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.02), 
                    bias_initializer = initializers.Zeros(),
                    activation = activations.tanh   
                        ))

    ann_model.add(Dense( units = 5, 
                    kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05),
                    bias_initializer = initializers.Zeros(),
                    activation = activations.relu
                        ))
    ann_model.add(Dense(1, kernel_initializer='normal'))
    ann_model.compile(loss='mean_squared_error', optimizer= optimizers.Adam(learning_rate = 0.01))
    ann_model.fit(x_train, y_train ,batch_size = 10, epochs = 30, verbose=0)
    curr_epoch = time.time()
    ann_model.save(os.path.join(DIR, f"temp_ann_{curr_epoch}.h5"), save_format='h5')
    print('  ')
    print("Newly trained ANN Model saved as: ", os.path.join(DIR, f"temp_ann_{curr_epoch}.h5"))
    print('  ')

    #Inverse transform of Test Data
    global PredictorScaler, TargetScaler
    predict = TargetScaler.inverse_transform(ann_model.predict(x_test))
    target_orig = TargetScaler.inverse_transform(y_test)

    df = pd.DataFrame(data = predict, columns = ['temperature_pred'])
    df['temperature_original'] = target_orig
    # print()
    df['submergence'] = PredictorScaler.inverse_transform(x_test)[:, 1]
    df.sort_values(by = ['submergence'], ascending=True)
    print(df)

    print('    ------- ##### -------     ')


def main():
    fire.Fire()

if __name__ == '__main__':
    main()

print('    ')

