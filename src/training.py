import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from models.emotionvgg import EmotionVGGDefault
from dataloader.fer2013dataloader import Fer2013BasicGenerator
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

earlystop = EarlyStopping(monitor="val_loss",
                        min_delta=0,
                        patience=10,
                        verbose=1,
                        restore_best_weights=True
                        )


reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                factor=0.2,
                                patience=5,
                                verbose=1,
                                min_delta=0.0001)


def get_tensorboard_cb():
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=5, histogram_freq=1, write_graph=True, write_images=True)
    return tbCallBack

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("confmatrix3_mine.png")
    return ax

def get_classweights(classes,normalize = False):
    counter = Counter(classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 
    
    if normalize:
        factor=1.0/sum(class_weights.values())
        for k in class_weights:
            class_weights[k] = class_weights[k]*factor
    return class_weights


class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
logdir = "./Graph"
csv_path = "data/fer2013.csv"
batch_size = 32

generator_constructor = Fer2013BasicGenerator(csv_path=csv_path, batch_size=batch_size, input_size=(48, 48, 1))
training_generator, val_X, val_Y,test_X, test_Y = generator_constructor.get_generators()

#model, modelname = EmotionVGGDefault.build(width=48, height=48, depth = 1, classes = 7,last_activation="softmax")
#model.compile(optimizer="adam",loss="categorical_crossentropy", metrics="accuracy")

model, modelname = EmotionVGGDefault.build(width=48, height=48, depth = 1, classes = 7,last_activation="sigmoid")
model.compile(optimizer="adam",loss="binary_crossentropy", metrics="accuracy")

# PREPARE TRAIN
ys = training_generator.y
ys2 = np.argmax(ys, axis=1)
class_weights_dict = get_classweights(ys2,normalize=False)
STEP_SIZE_TRAIN= training_generator.n// training_generator.batch_size
epochs = 30

# TRAIN
model.fit(training_generator,steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=(val_X, val_Y),
                                class_weight=class_weights_dict,
                                epochs=epochs, 
                                #callbacks=[get_tensorboard_cb(),earlystop,reduce_lr],
                                verbose=1)

# TEST
pred = model.predict(test_X)
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(test_Y, axis=1)
matrix = confusion_matrix(y_true, y_pred)
print("CNN Model Accuracy on test set: {:.4f}".format(accuracy_score(y_true, y_pred)))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_labels))
plot_confusion_matrix(y_true,y_pred,classes=class_labels,normalize=True)

