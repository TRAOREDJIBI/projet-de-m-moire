# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pickle 

loaded_model = pickle.load(open('C:/Users/HP/OneDrive/Bureau/deploying/trained_model .sav', 'rb'))

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Conversion en tableau NumPy
input_data_as_numpy_array = np.asarray(input_data)

# Reshape des données
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Prédiction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

# Affichage du résultat
if prediction[0] == 0:
    print('The person is  diabetic')
else:
    print('The person is not diabetic')