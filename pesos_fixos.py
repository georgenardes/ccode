import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation
import cv2 as cv
import numpy as np
import tensorflow_model_optimization as tfmot


# define modelo
model = Sequential()
model.add(Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), input_shape=(6, 3, 1), padding='same'))
model.add(Activation('relu'))



# ATRIBUI PESOS AO MODELO
model.layers[0].weights[0].assign([[[[ 1]],
                                    [[ 2]],
                                    [[1]]],
                                  [[[0]],
                                    [[0]],
                                    [[0]]],
                                  [[[ -1]],
                                    [[-2]],
                                    [[-1]]]])

model.layers[0].weights[1].assign([0])
print(model.layers[0].weights[0])
print(model.layers[0].weights[1])

# DEFINE IMAGEM DE ENTRADA
input_im = np.asarray([[[[32],[32],[32]],[[0],[0],[0]],[[0],[0],[0]],[[64],[64],[32]],[[0],[0],[0]],[[0],[0],[0]]]])
print("IMAGEM DE ENTRADA", input_im.shape)
print(input_im)

# EXECUTA O MODELO COM A IMAGEM DE ENTRADA E SALVA RESULTADO
output_im = model(input_im).numpy()
print("RESULTADO", output_im.shape)
print(output_im)

# cria objeto para quantizacao
quantize_model = tfmot.quantization.keras.quantize_model

# CRIA MODEO PARA QUANTIZAÇÃO
q_aware_model = quantize_model(model)
q_aware_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[tf.keras.metrics.MeanSquaredError()])


# TREINA O MODELO QUANTIZADO PARA A IMAGEM DE ENTRADA E O RESULTADO OBTIDO ANTERIORMENTE
q_aware_model.fit(x=input_im, y=output_im, epochs=5000)

# CONVERTE O MODELO PARA REPRESENTACAO DO TFLITE
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# CRIA INTERPRETADOR DO TFLITE
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

# OBTEM INDICE DE ENTRADA E SAIDA DO INTEPRETADOR
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Pre-processing: add batch dimension and convert to float32 to match with
# the model's input data format.
# test_image = np.expand_dims(X_train[0], axis=0).astype(np.float32)
test_image = input_im.astype(np.float32)
interpreter.set_tensor(input_index, test_image)

# Run inference.
interpreter.invoke()

# Post-processing: remove batch dimension and find the digit with highest
# probability.
output = interpreter.tensor(output_index)


# APRESENTA OS DETALHES DAS OPERACOES DO INTERPRETADOR
for od in interpreter._get_ops_details():
  print(od)

'''
{'index': 1, 'op_name': 'CONV_2D', 'inputs': ARRAY DE INDICES DOS TENSORES QUE SÃO ENTRADA DESSA OPERACAO, 'outputs': ARRAY DE INDICES DOS TENSORES QUE SÃO SAIDA DESSA OPERACAO}
'''


print("DADO DE ENTRADA \n=================================\n0")
print(interpreter.tensor(0)())
print(interpreter.get_tensor_details()[0])
print("=================================")
print("DADO DE ENTRADA QUANTIZADO \n=================================\n2")
print(interpreter.tensor(3)())
# s, z = interpreter.get_tensor_details()[3]['quantization']
# r = (q - z) * s
# q = (r / s) + z
# print((interpreter.tensor(3)().astype(np.int32) - z) * s)
print(interpreter.get_tensor_details()[3])
print("=================================")
print("DADO DE SAIDA DA CONV \n=================================\n4")
print(interpreter.tensor(4)())

# DEQUANTIZACAO MANUAL COM OS PARAMETROS s z DE QUANTIZAÇÃO
# s, z = interpreter.get_tensor_details()[4]['quantization']
# print((interpreter.tensor(4)().astype(np.int32) - z) * s)

print(interpreter.get_tensor_details()[4])
print("=================================")
print("DADO DE SAIDA DA DEQUANTIZAÇÃO \n=================================\n5")
print(interpreter.tensor(5)())
print(interpreter.get_tensor_details()[5])
print("=================================")



print("PESOS QUANTIZADO DURANTE TREINAMENTO \n=================================\n")
print(interpreter.tensor(2)())
print(interpreter.get_tensor_details()[2])
print("=================================")
print("BIAS QUANTIZADO \n=================================\n")
print(interpreter.tensor(1)())
print(interpreter.get_tensor_details()[1])
print("=================================")

print("saida real", output_im)
