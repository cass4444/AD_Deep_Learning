import numpy as np
import pandas as pd
import csv
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint

# Image_Classifier Function
# TODO Update classifier using ResNet50 and Inception
# TODO Calculate metrics: accuracy, sensitiviy, recall, performance


def image_classifier():
    num_classes = 2
    classes = ['AD', 'not_AD']
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S

    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    save_date = now.strftime("%m_%d_%y_%H_%M_%S")
    save_fig_path = Path('results/Graphs') / (save_date + '_AD.png')
    save_cm_path = Path('results/Confusion_Matrix') / (save_date + '.png')
    np.random.seed(seed=7)
    for test_epochs in range(1, 2, 3):
        my_new_model = Sequential()
        my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
        my_new_model.add(Dense(num_classes, activation='softmax'))

        # Indicate whether the first layer should be trained/changed or not.
        my_new_model.layers[0].trainable = False

        print("Specify Model Complete")

        my_new_model.compile(optimizer='sgd',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        print("Compile Model Complete")

        # checkpoint
        filepath = "weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        image_size = 299
        seed = 7
        data_generator = ImageDataGenerator(preprocess_input, validation_split=0.2)
        aug_data_generator = ImageDataGenerator(preprocess_input,
                                                validation_split=0.2,
                                                horizontal_flip=True,
                                                vertical_flip=True)
        path = Path("images")

        train_generator = aug_data_generator.flow_from_directory(
            directory=path,
            target_size=(image_size, image_size),
            batch_size=10,
            class_mode='categorical',
            subset="training",
            seed=seed)

        validation_generator = data_generator.flow_from_directory(
            directory=path,
            target_size=(image_size, image_size),
            batch_size=10,
            class_mode='categorical',
            subset="validation",
            seed=seed)
        print(train_generator.class_indices)
        print(validation_generator.class_indices)
        labels = validation_generator.class_indices
        print(labels)
        labels = dict((v, k) for k, v in labels.items())
        print(labels)
        filenames = validation_generator.filenames
        true_labels = validation_generator.classes
        print(filenames)
        print(true_labels)
        validation_generator.reset()
        # fit_stats below saves some statistics describing how model fitting went
        # the key role of the following line is how it changes my_new_model by fitting to data
        fit_stats = my_new_model.fit_generator(train_generator,
                                               steps_per_epoch=17,
                                               epochs=test_epochs,
                                               validation_data=validation_generator,
                                               validation_steps=5,
                                               # callbacks=callbacks_list,
                                               verbose=1)

        validation_generator.reset()
        pred = my_new_model.predict(validation_generator, steps=5, verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        truth = [labels[t] for t in true_labels]
        predictions = [labels[k] for k in predicted_class_indices]
        results = pd.DataFrame({"Filenames": filenames, "Truth": truth, "Predictions": predictions})
        predictions_csv_path = Path("results/Predictions/predictions.csv")
        results.to_csv(predictions_csv_path, index=False)
        plot_confusion_matrix(truth, predictions, classes=classes, save_cm_path=save_cm_path,
                              title='Confusion matrix, without normalization')
        # plt.show()

        save_result_path = Path('results/Tables/result_log.csv')
        print(save_fig_path)
        print("date and time =", dt_string)
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(fit_stats.history['acc'])
        axs[0].plot(fit_stats.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend(['Train', 'Test'], loc='upper left')
        fig.suptitle('Model Run ' + dt_string, fontsize=16)

        axs[1].plot(fit_stats.history['loss'])
        axs[1].plot(fit_stats.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend(['Train', 'Test'], loc='upper left')
        plt.savefig(save_fig_path)
        # plt.show()

        val_acc = fit_stats.history['val_acc']
        acc = fit_stats.history['acc']
        val_loss = fit_stats.history['val_loss']
        loss = fit_stats.history['loss']
        with open(save_result_path, 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([dt_string]+val_acc+acc+val_loss+loss)
        csvFile.close()

        print("Fit Model Complete")


# U-Net Semantic Segmentation Function
# TODO Create U-Net Lesion Semantic Segmentator
# def u_net_lesion():


def plot_confusion_matrix(y_true, y_pred, classes, save_cm_path,
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
    cm = confusion_matrix(y_true, y_pred, classes)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
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
    plt.savefig(save_cm_path)
    return ax


image_classifier()
