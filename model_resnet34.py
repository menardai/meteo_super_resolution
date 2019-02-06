import fastai

from fastai.vision import *
from fastai.callbacks import *
from fastprogress import force_console_behavior

from weather_dataset import Weather3to3


def unet_resnet34(data_loader, weights_filename=None):
    """
    Returns a unet learner based on a Resnet-34 as encoder.
    The encoder takes as input and generate a 3 channels RGB PIL Image.
    """
    learn_gen = create_gen_learner(data_loader, models.resnet34)

    if weights_filename is not None:
        learn_gen.load(weights_filename)

    return learn_gen


def create_gen_learner(data_loader, architecture):
    """
    Create a unet learner based on the specified encoder architecture using the given data loader.
    """
    wd = 1e-3  # weight decay
    y_range = (-3.,3.)
    loss_gen = MSELossFlat()

    return unet_learner(data_loader, architecture,
                        wd=wd, blur=True, norm_type=NormType.Weight,
                        self_attention=True, y_range=y_range, loss_func=loss_gen)


def get_temperature_vector(channel_vector, min_value, max_value):
    """
    Returns a (256,256) float vector ranging from [min_value, max_value].

    Arguments:
        channel_vector: a channel with values between [0, 1]
        min_value: integer, the minimum value of the entire distribution of the vector to return
        max_value: integer, the maximum value of the entire distribution of the vector to return
    """
    return channel_vector * (max_value - min_value) + min_value


def save_predictions(learn_gen, data_loader, date_indexes, png_output_folder, output_array_filename=None):
    filenames = data_loader.dataset.items
    i=0

    # make output folder for all png, if not exist
    path_gen = Path(png_output_folder)
    path_gen.mkdir(exist_ok=True)

    if output_array_filename is not None:
        temperature_maps = np.zeros(shape=(len(filenames), 1, 256, 256))

    for b in data_loader:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)

        for output_image in preds:
            # save png image on disk
            output_image.save(path_gen/filenames[i].name)

            if output_array_filename is not None:
                index = np.where(date_indexes == str.encode(Path(filenames[i]).name[:-4]))

                # transpose from fastai images shape of (c, h, w) to (h, w, c)
                output_channels = output_image.data.permute(1, 2, 0)
                # convert the red channel to temperature values and store in the array to be saved
                temperature_maps[index,0,:,:] = get_temperature_vector(output_channels[:,:,0], -50, 50)

            i += 1

    if output_array_filename is not None:
        # save to a .npy file
        np.save(output_array_filename, temperature_maps)
        return temperature_maps

    return None


def train():
    # create the dataset loader
    data_loader = Weather3to3('./images/', batch_size=34, image_size=256).data_loader

    # create the unet architecture with a Resnet-34 as encoder
    learn_gen = unet_resnet34(data_loader)

    # by default the Resnet encoder weights are frozen to Imagenet pre-trained weights,
    # keep them freeze and train only for 2 epochs to train the rest of the unet
    learn_gen.fit_one_cycle(2)

    # now, unfreeze and train all layers
    learn_gen.unfreeze()

    # fit the model on 300 epochs.
    # learning rate to cycle between 1e-6 and 1e-3
    learn_gen.fit_one_cycle(300, slice(1e-6,1e-3))

    learn_gen.save('unet_resnet34')

    date_indexes = np.load('data/date_test_set.npy')
    save_predictions(learn_gen,
                     data_loader.valid_dl,
                     date_indexes,
                     'images/valid/image_gen_34',
                     output_array_filename='images/image_gen_34_test.npy')


def predict():
    data_loader = Weather3to3('./images/', batch_size=8, image_size=256).data_loader

    # create the unet architecture with a Resnet-34 as encoder
    learn_gen = unet_resnet34(data_loader, weights_filename='unet_resnet34')

    date_indexes = np.load('data/date_test_set.npy')
    save_predictions(learn_gen,
                     data_loader.valid_dl,
                     date_indexes,
                     'images/valid/image_gen_34p',
                     output_array_filename='images/image_gen_34p_test.npy')


if __name__ == "__main__":
    """
    Arguments:
      --predict
      --train
    """
    master_bar, progress_bar = force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

    print('------------------------------------------')
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == '--predict':
        predict()

    elif arg == '--train':
        train()

    else:
        print('Usage:')
        print('  --train')
        print('  --predict')
