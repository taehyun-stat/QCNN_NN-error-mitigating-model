# Implementation of Quantum Convolutional Neural Network (QCNN) circuit structure.

import pennylane as qml
import unitary
import embedding

# Convolutional layers (1st, 2nd, 3rd layers)
def conv_layer1(U, params):
    U(params, wires=[7, 0])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])
def conv_layer2(U, params):
    U(params, wires=[6, 0])  
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])
def conv_layer3(U, params):
    U(params, wires=[4,0])

# Pooling layers (1st, 2nd, 3rd layers)
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])
def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])
def pooling_layer3(V, params):
    V(params, wires=[4,0])


# QCNN circuit structure build with pooling layers
def QCNN_structure(U, params, U_params):
    # parameter allocation (convolutional layers and pooling layers)
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]
    
    # parameters of the convolutional layers and pooling layers
    conv_layer1(U, param1)
    pooling_layer1(Pooling_ansatz1, param4)

    conv_layer2(U, param2)
    pooling_layer2(Pooling_ansatz1, param5)

    conv_layer3(U, param3)
    pooling_layer3(Pooling_ansatz1, param6)

# QCNN circuit structure build without pooling layers
def QCNN_structure_without_pooling(U, params, U_params):
    # parameter allocation (convolutional layers and pooling layers)
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    
    # parameters of the convolutional layers
    conv_layer1(U, param1)
    
    conv_layer2(U, param2)
    
    conv_layer3(U, param3)


def QCNN_1D_circuit(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2,3])
    U(param2, wires=[4,5])

    U(param3, wires=[3,4])


# Set up the quantum circuit from PennyLane
dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)

# QCNN circuit build with data encoding, convolutional layers, pooling layers, and measurement
def QCNN(X, params, U, U_params, embedding_type='Amplitude', cost_fn='cross_entropy', measure_axis='Z'):

    # Data Embedding
    data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(U_TTN, params, U_params)
    elif U == 'U_9':
        QCNN_structure(U_9, params, U_params)
    elif U == 'U_15':
        QCNN_structure(U_15, params, U_params)
    elif U == 'U_13':
        QCNN_structure(U_13, params, U_params)
    elif U == 'U_14':
        QCNN_structure(U_14, params, U_params)
    elif U == 'U_SO4':
        QCNN_structure(U_SO4, params, U_params)        
    elif U == 'U_5':
        QCNN_structure(U_5, params, U_params)
    elif U == 'U_6':
        QCNN_structure(U_6, params, U_params)
    elif U == 'U_SU4':
        QCNN_structure(U_SU4, params, U_params)
    elif U == 'U_SU4_no_pooling':
        QCNN_structure_without_pooling(U_SU4, params, U_params)
    elif U == 'U_9_1D':
        QCNN_1D_circuit(U_9, params, U_params)
    elif U == 'U_SU4_1D':
        QCNN_1D_circuit(U_SU4, params, U_params)

    else:
        print("Invalid Unitary Ansatze")
        return False
    
    # Measurement in Z bases(computaional bases)
    if measure_axis == 'Z':
        if cost_fn == 'mse' or cost_fn == 'mse_lambda' or cost_fn == 'classify_lambda' or cost_fn == 'classify_lambda2':
            result = qml.expval(qml.PauliZ(0))
        elif cost_fn == 'cross_entropy':
            result = qml.probs(wires=0)
        return result

    # Measurement in X bases(Hadamard bases)
    if measure_axis == 'X':
        if cost_fn == 'mse' or cost_fn == 'mse_lambda' or cost_fn == 'classify_lambda' or cost_fn == 'classify_lambda2':
#             qml.Hadamard(wires=0)
#             result = qml.expval(qml.PauliZ(0))
            result = qml.expval(qml.PauliX(0))
        elif cost_fn == 'cross_entropy':
            qml.Hadamard(wires=0)
            result = qml.probs(wires=0)
        return result

    # Measurement in Y bases(circular bases)
    if measure_axis == 'Y':
        if cost_fn == 'mse' or cost_fn == 'mse_lambda' or cost_fn == 'classify_lambda' or cost_fn == 'classify_lambda2':
#             qml.adjoint(qml.S(wires=0))
#             qml.Hadamard(wires=0)
#             result = qml.expval(qml.PauliZ(0))
            result = qml.expval(qml.PauliY(0))
        elif cost_fn == 'cross_entropy':
            qml.adjoint(qml.S(wires=0))
            qml.Hadamard(wires=0)
            result = qml.probs(wires=0)
        return result
    