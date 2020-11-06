# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:50:40 2020

@author: nnig9
"""
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, LSTMCell, RNN, Bidirectional, concatenate
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json


data=pd.read_csv('cl.csv', sep=';', error_bad_lines=False)
data['Bid'] = data['Bid'].apply(lambda x: float(x))
data['date'] = pd.to_datetime(data['date'])

########      Фильтры данных
########      Фильтр по размеру
########      [768,  320,    1,  480,  728,  300, 1440, 1024,  640]
########      [1024,  480,    1,  320,  800,   50,   90,  250,  300, 1080,  768, 100]
########      ['simple', 'native', 'dhtml', 'video']

b = 'simple'
banner_width = 728
banner_heigth = 90
start= '2020-10-01'
end = '2020-10-08'

##############################################
######    преобразование данных
#№№№№№    разбивка на дату и время

data['dat'] = [d.date() for d in data['date']]
data['time'] = [d.time() for d in data['date']]

#####    Удаление столбцов и изменеие порядка

data.drop('Click_ID',axis = 1,inplace=True)
data.drop('date',axis = 1,inplace=True)
data = data[['dat',
             'time',
             'banner_type',
             'banner_width',
             'banner_heigth',
             'IP','Device_ID',
             'Bid','OS',
             'OS_Version',
             'Device_mode',
             'Connection_Type',]]

########  Фильтр по дате

#start= '2020-10-01'
start =pd.to_datetime(start).date()
#end = '2020-10-08'
end =pd.to_datetime(end).date()
data[data.dat.between(start, end)]



########      Фильтр по типу
########      ['simple', 'native', 'dhtml', 'video']
#b = 'simple'
data = data.query('banner_type == @b' )

########      Фильтр по размеру
########     [ 768,  320,    1,  480,  728,  300, 1440, 1024,  640]
########     [1024,  480,    1,  320,  800,   50,   90,  250,  300, 1080,  768, 100]

data = data.query('banner_width == 728 and banner_heigth == 90')

########      Размер дф
lenn = len(data)
print('Размер= ', lenn)


series = data['Bid'][-3000:].reset_index()
series = series.drop('index',axis=1)
seq = series.copy()
scaler = MinMaxScaler(feature_range=(-1,1))
X =scaler.fit_transform(seq.values.reshape(-1,1))

###################################################
####################### ГРАФИК
series = X
seq = series.copy()
plt.figure(figsize=(15,7))
plt.title("Sequence", fontsize = 18)
plt.xlabel("Time", fontsize = 18)
plt.ylabel("Bid", fontsize = 18)
plt.plot(seq)
plt.show()

####################### Параметры сети
input_seq_len = 100
output_seq_len = 30
n_in_features = 1
n_out_features = 1
batch_size = 10


x_train = X[:-input_seq_len]
x_test = X[-input_seq_len:]


total_loss = []
total_val_loss = []

def generate_train_sequences(x):
    
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), total_start_points, replace = False)
    
    input_batch_idxs = [(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(x, output_batch_idxs, axis = 0)
    
    input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],n_in_features))
    output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],n_out_features))
    
    return input_seq, output_seq

def create_model(layers, bidirectional = False):
    
    n_layers = len(layers)
    
    ## Encoder
    encoder_inputs = Input(shape = (None, n_in_features))
    lstm_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]
    
    if bidirectional:
        
        encoder = Bidirectional(RNN(lstm_cells, return_state=True))
        encoder_outputs_and_states = encoder(encoder_inputs)
        bi_encoder_states = encoder_outputs_and_states[1:]
        encoder_states = []
        
        for i in range(int(len(bi_encoder_states) / 2)):
            
            temp = []
            for j in range(2):
                
                temp.append(concatenate([bi_encoder_states[i][j], bi_encoder_states[n_layers + i][j]], axis = -1))
                
            encoder_states.append(temp)
    else:  
        
        encoder = RNN(lstm_cells, return_state = True)
        encoder_outputs_and_states = encoder(encoder_inputs)
        encoder_states = encoder_outputs_and_states[1:]
    
    ## Decoder
    decoder_inputs = Input(shape = (None, n_out_features))
    
    if bidirectional:
        
        decoder_cells = [LSTMCell(hidden_dim*2) for hidden_dim in layers]
    else:
        
        decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]
        
    decoder_lstm = RNN(decoder_cells, return_sequences = True, return_state=True)
    decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state = encoder_states)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_dense = Dense(n_out_features) 
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
    return model

def run_model(model,batches,epochs,batch_size):

    for _ in range(batches):

        input_seq, output_seq = generate_train_sequences(x_train)

        encoder_input_data = input_seq
        decoder_target_data = output_seq
        decoder_input_data = np.zeros(decoder_target_data.shape)

        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_split=0.1, 
                             shuffle=False)
                           
        total_loss.append(history.history['loss'])
        total_val_loss.append(history.history['val_loss'])


model2_bi = create_model([input_seq_len,input_seq_len],bidirectional=True)

# # загрузка модели
# json_file = open('model2_bi.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model2_bi = model_from_json(loaded_model_json)
# ## загрузка весов
# model2_bi.load_weights("model2_bi.h5")
# print("Loaded model from disk")


model2_bi.compile(Adam(), loss = 'mean_squared_error')
start_time = time.time()
#print(model2_bi.summary())

#================ОБУЧЕНИЕ======================================================
run_model(model2_bi,batches=1, epochs=2, batch_size=batch_size)

end_time = time.time()
run_time = (end_time - start_time)/60
print("Время работы - ",run_time)

total_loss = [j for i in total_loss for j in i]
total_val_loss = [j for i in total_val_loss for j in i]

#График обучения
# plot_loss(total_loss,total_val_loss)
# plt.savefig('bi_los2.png')

input_seq_test = x_train[-input_seq_len:].reshape((1,input_seq_len,1))
output_seq_test = x_test[:output_seq_len]
decoder_input_test = np.zeros((1,output_seq_len,1))

print(x_train[-input_seq_len:])
#------------------------------------------------------------------------------
pred2_bi = model2_bi.predict([input_seq_test,decoder_input_test])

pred_values2_bi = scaler.inverse_transform(pred2_bi.reshape(-1,1))
output_seq_test2_bi = scaler.inverse_transform(output_seq_test)

print("Прогноз на день")
print(pred_values2_bi)

#рисунок прогноза
plt.plot(pred_values2_bi, label = "pred")
plt.plot(output_seq_test2_bi, label = "actual")
plt.title("Prediction vs Actual")
plt.ylabel("Bid", fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.legend()
plt.savefig('bi_dir2.png')

# сохранение модели в JSON
# model_json = model2_bi.to_json()
# with open("model2_bi.json", "w") as json_file:
#     json_file.write(model_json)
# # сохранение весов в HDF5
# model2_bi.save_weights("model2_bi.h5")
# print("Saved model to disk")
