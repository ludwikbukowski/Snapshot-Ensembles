import json
import numpy as np
import argparse
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from models import wide_residual_net as WRN, dense_net as DN
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from mygen import BatchGenerator
import keras
import keras.utils.np_utils as kutils

parser = argparse.ArgumentParser(description='CIFAR 10 Ensemble Prediction')

parser.add_argument('--optimize', type=int, default=0, help='Optimization flag. Set to 1 to perform a randomized '
                                                            'search to maximise classification accuracy.\n'
                                                            'Set to -1 to get non weighted classification accuracy')

parser.add_argument('--num_tests', type=int, default=20, help='Number of tests to perform when optimizing the '
                                                              'ensemble weights for maximizing classification accuracy')

parser.add_argument('--model', type=str, default='wrn', help='Type of model to train')

# Wide ResNet Parameters
parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')

# DenseNet Parameters
parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')

args = parser.parse_args()

# Change NUM_TESTS to larger numbers to get possibly better results
NUM_TESTS = args.num_tests

# Change to False to only predict
OPTIMIZE = args.optimize

model_type = str(args.model).lower()
assert model_type in ['wrn', 'dn'], 'Model type must be one of "wrn" for Wide ResNets or "dn" for DenseNets'

if model_type == "wrn":
    n = args.wrn_N * 6 + 4
    k = args.wrn_k

    models_filenames = [
        r"weights/WRN-CIFAR10-%d-%d-Best.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-1.h5" % (n, k),
                        # r"weights/WRN-CIFAR10-%d-%d-2.h5" % (n, k),
                        # r"weights/WRN-CIFAR10-%d-%d-3.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-4.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-5.h5" % (n, k)
                        # r"weights/WRN-CIFAR10-%d-%d-6.h5" % (n, k)
                        ]
else:
    depth = args.dn_depth
    growth_rate = args.dn_growth_rate

    models_filenames = [
        r"weights/DenseNet-CIFAR10-%d-%d-Best.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-1.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-2.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-3.h5" % (depth, growth_rate),
                        # r"weights/DenseNet-CIFAR10-%d-%d-2.h5" % (depth, growth_rate),
                        # r"weights/DenseNet-CIFAR10-%d-%d-3.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-4.h5" % (depth, growth_rate)
                        # r"weights/DenseNet-CIFAR10-%d-%d-6.h5" % (depth, growth_rate)
                        ]

(trainX, trainY), (testX, testY) = cifar10.load_data()
nb_classes = len(np.unique(testY))

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY_cat = kutils.to_categorical(trainY)
testY_cat = kutils.to_categorical(testY)

if K.image_dim_ordering() == "th":
    init = (3, 32, 32)
else:
    init = (32, 32, 3)

if model_type == "wrn":
    model = WRN.create_wide_residual_network(init, nb_classes=10, N=args.wrn_N, k=args.wrn_k, dropout=0.00)

    model_prefix = 'WRN-CIFAR10-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
else:
    model = DN.create_dense_net(nb_classes=10, img_dim=init, depth=args.dn_depth, nb_dense_block=1,
                                growth_rate=args.dn_growth_rate, nb_filter=16, dropout_rate=0.2)

    model_prefix = 'DenseNet-CIFAR10-%d-%d' % (args.dn_depth, args.dn_growth_rate)

best_acc = 0.0
best_weights = None

# train_preds = []
# for fn in models_filenames:
#     model.load_weights(fn)
#     print("Predicting train set values on model %s" % (fn))
#     yPreds = model.predict(trainX, batch_size=128, verbose=2)
#     train_preds.append(yPreds)

epochs=3
batch_size=32
save_dir="supernet"




members = []
test_preds = []
for fn in models_filenames:
    model.load_weights(fn)
    print("Predicting test set values on model %s" % (fn))

    ## My custom
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])
    # yPreds = model.predict(testX, batch_size=128, verbose=2)
    # scores = model.evaluate(testX, testY_cat, verbose=1)
    mymodel = keras.models.clone_model(model)
    mymodel.set_weights(model.get_weights())
    members.append(mymodel)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])

    # test_preds.append(yPreds)

class Predictor:

    def __init__(self, model):
        self._model = model
        self._first_layer = model.get_layer(index=0).input

    def predict(self, x, index=-1):
        first_layer = self._first_layer
        last_layer = self._model.get_layer(index=index).output
        print(last_layer.name)
        new_model = Model(inputs=[first_layer], outputs=[last_layer])
        return new_model.predict(x)


def create_sets(members, x_train, x_test):
    predictors = tuple(Predictor(model) for model in members)
    new_training_set = np.concatenate([predictor.predict(x_train, -2) for predictor in predictors], axis=1)
    new_testing_set = np.concatenate([predictor.predict(x_test, -2) for predictor in predictors], axis=1)
    return (new_training_set, new_testing_set)

def define_supernet(members, new_training_set):
    model = Sequential([
        Dense(num_classes, activation='softmax', input_shape=(new_training_set.shape[1],))
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    all_weights = [m.get_layer(index=-1).get_weights() for m in members]

    weights = np.concatenate([w for w, _ in all_weights], axis=0)
    biases = np.mean([b for _, b in all_weights], axis=0)

    model.get_layer(index=-1).set_weights([weights, biases])
    return model

x_train_new, x_test_new = create_sets(members, trainX, testX)
supernet = define_supernet(members, x_train_new)

scores = supernet.evaluate(x_test_new, testY, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

csv_logger = CSVLogger(save_dir + '/history.csv', append=True, separator=';')
supernet.fit(x_train_new, trainY,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test_new, testY),
          callbacks=[csv_logger])

scores = supernet.evaluate(x_test_new, testY, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


def calculate_weighted_accuracy():
    global weighted_predictions, weight, prediction, yPred, yTrue, accuracy, error
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, test_preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
    exit()


if OPTIMIZE == 0:
    with open('weights/Ensemble weights %s.json' % model_prefix, mode='r') as f:
        dictionary = json.load(f)

    prediction_weights = dictionary['best_weights']
    calculate_weighted_accuracy()

elif OPTIMIZE == -1:
    prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
    calculate_weighted_accuracy()

''' OPTIMIZATION REGION '''

print()

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((trainX.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(weights, train_preds):
        final_prediction += weight * prediction

    return log_loss(trainY_cat, final_prediction)


for iteration in range(NUM_TESTS):
    prediction_weights = np.random.random(len(models_filenames))

    constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(train_preds)

    result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    print('Best Ensemble Weights: {weights}'.format(weights=result['x']))

    weights = result['x']
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(weights, test_preds):
        weighted_predictions += weight * prediction

    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
    print("Iteration %d: Error : " % (iteration + 1), error)

    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights

    print()

print("Best Accuracy : ", best_acc)
print("Best Weights : ", best_weights)

with open('weights/Ensemble weights %s.json' % model_prefix, mode='w') as f:
    dictionary = {'best_weights' : best_weights.tolist()}
    json.dump(dictionary, f)

''' END OF OPTIMIZATION REGION '''

