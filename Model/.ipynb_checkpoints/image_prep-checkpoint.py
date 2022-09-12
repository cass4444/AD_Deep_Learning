from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

def image_prep():

    image_size = 224
    image_generator1 = ImageDataGenerator(
        # rotation_range=40,
        # width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        preprocessing_function=preprocess_input,
        validation_split=0.1)

    image_path = Path("images")

    train_val_generator = image_generator1.flow_from_directory(
        directory=image_path,
        target_size=(image_size, image_size),
        batch_size=10,
        class_mode='categorical',
        save_to_dir='aug_train_val',
        save_prefix='aug_train_val',
        save_format='jpeg',
        subset='training',
        seed=7)

    test_generator = image_generator1.flow_from_directory(
        directory=image_path,
        target_size=(image_size, image_size),
        batch_size=10,
        class_mode='categorical',
        save_to_dir='aug_test',
        save_prefix='aug_test',
        save_format='jpeg',
        subset='validation',
        seed=7)
    print(train_val_generator.samples)
    print(test_generator.samples)


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def example():
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    img = load_img('images/AD/atopic1.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='aug_test', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


# example()
image_prep()



