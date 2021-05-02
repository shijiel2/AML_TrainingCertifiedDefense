from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np
import os
import dataaug
import timeit
import argparse

from art.attacks.poisoning.perturbations.image_perturbations import (
    add_pattern_bd,
    add_single_bd,
)
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, preprocess
from art.defences.detector.poison import ActivationDefence


def generate_backdoor(
    x_clean,
    y_clean,
    percent_poison,
    backdoor_type="pattern",
    sources=np.arange(10),
    targets=(np.arange(10) + 1) % 10,
):
    """
    Creates a backdoor in MNIST images by adding a pattern or pixel to the image and changing the label to a targeted
    class. Default parameters poison each digit so that it gets classified to the next digit.
    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param sources: Array that holds the source classes for each backdoor. Poison is
    generating by taking images from the source class, adding the backdoor trigger, and labeling as the target class.
    Poisonous images from sources[i] will be labeled as targets[i].
    :type sources: `np.ndarray`
    :param targets: This array holds the target classes for each backdoor. Poisonous images from sources[i] will be
                    labeled as targets[i].
    :type targets: `np.ndarray`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, x_poison, which
    contains all of the data both legitimate and poisoned, and y_poison, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """

    max_val = np.max(x_clean)

    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison))

    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
        src_imgs = x_clean[y_clean == src]

        n_points_in_src = np.shape(src_imgs)[0]
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])
        if backdoor_type == "pattern":
            imgs_to_be_poisoned = add_pattern_bd(
                x=imgs_to_be_poisoned, pixel_value=max_val
            )
        elif backdoor_type == "pixel":
            imgs_to_be_poisoned = add_single_bd(
                imgs_to_be_poisoned, pixel_value=max_val
            )
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, np.ones(num_poison) * tgt, axis=0)
        is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="0")
    parser.add_argument("--end", default="1000")
    parser.add_argument("--k", default="30")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--poison_size", default="100")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = float(args.gpum)
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))

    batch_size = 16
    num_classes = 10
    epochs = 200

    # input image dimensions
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    perc_poison = float(args.poison_size) / len(x_train)
    (is_poison_train, x_train, y_train) = generate_backdoor(
        x_train, y_train, perc_poison
    )

    # Shuffle training data so poison is not together
    n_train = np.shape(y_train)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    input_shape = x_train.shape[1:]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    ## here we use the same initialization for all models, and you can also use different initialization for different models
    weights_initialize = model.get_weights()

    ### the parameter k is the number of training examples sampled from the training dataset to train each base model
    k_value = int(args.k)

    """ 
    track the label frequency for each testing input, and the last dimension is used to save the true label, 
    which is further used to compute the certified radius
    """
    aggregate_result = np.zeros([x_test.shape[0], num_classes + 1], dtype=np.int)

    ## data augmentation function
    datagen = dataaug.DataGeneratorFunMNIST()

    for repeat_time in range(int(args.start), int(args.end)):
        # sampling with replacement.
        print("*****************")
        print("Train base model:", repeat_time)

        sample_index = np.random.choice(x_train.shape[0], k_value, replace=True)
        x_train_sample = x_train[sample_index, :, :, :]
        y_train_sample = y_train[sample_index, :]

        # train the model
        starttraining_time = timeit.default_timer()
        model.fit_generator(
            datagen.flow(x_train_sample, y_train_sample, batch_size=batch_size),
            epochs=epochs,
            verbose=0,
            workers=4,
        )
        endtraining_time = timeit.default_timer()
        print("training time:", endtraining_time - starttraining_time)

        # # evaluate the base model and you can also comment it without influencing the results.
        # score = model.evaluate(x_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

        prediction_label = np.argmax(model.predict(x_test), axis=1)
        aggregate_result[np.arange(0, x_test.shape[0]), prediction_label] += 1

        # reinitialize the model, note that you can also use different parameters to initialize the model
        model.set_weights(weights_initialize)
    aggregate_result[np.arange(0, x_test.shape[0]), -1] = np.argmax(y_test, axis=1)

    ### save the results
    tmp_folder = "./results"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    tmp_folder += "/aggregate_result"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    tmp_folder += "/mnist"
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    aggregate_folder = tmp_folder + "/k_" + args.k + "_poison-size_" + args.poison_size
    if not os.path.exists(aggregate_folder):
        os.makedirs(aggregate_folder)
    np.savez(
        aggregate_folder
        + "/aggregate_batch_k_"
        + args.k
        + "_start_"
        + args.start
        + "_end_"
        + args.end
        + ".npz",
        x=aggregate_result,
    )


if __name__ == "__main__":
    main()
