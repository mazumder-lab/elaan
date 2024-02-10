from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

def create_model(input_shape=None, layers_dense=None, units=None, dropout=None):
    """Creates a fully connected neural network keras model.
    
    Args:
        input_shape: input shape of the keras model.
        layers_dense: number of intermediate layers, int scaler.
        units: dimension of units in the neural network, int scaler.
        dropout: fraction of units to drop, float scalar.
    
    Returns:
        model: Keras model instance.            
    """
    model = Sequential()
    # First layer
    model.add(Dense(units, activation='relu', input_shape=input_shape, name='input'))
    model.add(Dropout(dropout))

    # Intermediate layers
    for i in range(1, layers_dense):
        model.add(Dense(units, activation='relu', name='fully_connected_{}'.format(i)))
        model.add(Dropout(dropout))

    # Last layer
    model.add(Dense(1, activation='linear', name='output'))

    
    return model
