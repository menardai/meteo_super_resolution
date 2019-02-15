import fastai

from fastai.vision import *
from fastai.callbacks import *
from fastprogress import force_console_behavior

from weather_dataset import Weather3to3


class ComputeScoreOnEpochEnd(Callback):

    def __init__(self, learner, min_epoch_to_save=0):
        self.learner = learner
        self.min_epoch_to_save = min_epoch_to_save

        self.scores = []
        self.best_score = None

    def on_epoch_end(self, **kwargs):
        (success, score) = predict(learn_gen=self.learner, is_computing_score=True)

        self.scores.append(score)

        if self.best_score is None or score < self.best_score:
            self.best_score = score

            if len(self.scores) >= self.min_epoch_to_save:
                # got a new best score then save the model
                self.learner.save('unet_resnet34_best')
                self.save('./images/training_scores_best.npy')

        if self.best_score == score:
            print('score = {:2.4f} (best)'.format(score))
        else:
            print('score = {:2.4f}'.format(score))

        # Return True to stop training, False to continue
        return False

    def save(self, filename):
        np.save(filename, self.scores)


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


def save_predictions(learn_gen, data_loader, date_indexes, png_output_folder=None, output_array_filename=None):
    """
    Generate an increased-resolution image for each lowres image in the given data_loader.
    Each of those generated images will be saved in png format in the specified png_output_folder.
    Generated images are also converted in air temperature only data saved in the specified output_array_filename.

    Args:
        learn_gen (unet_learner object): the trained learner to use for making the prediction
        data_loader (dataset object): the data loader of the test set to load the input image from
        date_indexes (numpy array): the date of the test set in chronological order
        png_output_folder (string): folder path where the generated png files will be saved
        output_array_filename (string): filename of the file where the generated air temperature data will be saved
    """
    filenames = data_loader.dataset.items
    i=0

    # make output folder for all png, if not exist
    if png_output_folder is not None:
        path_gen = Path(png_output_folder)
        path_gen.mkdir(exist_ok=True)

    # create an empty maps to collect temperature for each sample
    if output_array_filename is not None:
        temperature_maps = np.zeros(shape=(len(filenames), 1, 256, 256))

    for b in data_loader:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)

        for output_image in preds:
            if png_output_folder is not None:
                # save png image on disk
                output_image.save(path_gen/filenames[i].name)

            if output_array_filename is not None:
                # find the index of the filename in the dates array
                index = np.where(date_indexes == str.encode(Path(filenames[i]).name[:-4]))

                # transpose from fastai images shape of (c, h, w) to (h, w, c)
                output_channels = output_image.data.permute(1, 2, 0)
                # convert the red channel to temperature values and store in the array to be saved
                temperature_maps[index,0,:,:] = get_temperature_vector(output_channels[:,:,0], -50, 50)

            i += 1

    if output_array_filename is not None:
        # save air temperatures to a .npy file in the same format as the input and target from the dataset
        np.save(output_array_filename, temperature_maps)
        return temperature_maps

    return None


def train():
    """
        Train the model from the dataset png images in "./images/" folder.

        The trained model weights will be save in "./images/models/unet_resnet34.pth"
        The training losses plot will be save in "./losses_plot.png"
    """
    # create the dataset loader
    data_loader = Weather3to3('./images/', batch_size=12, image_size=256).data_loader

    # create the unet architecture with a Resnet-34 as encoder
    learn_gen = unet_resnet34(data_loader)

    score_callback = ComputeScoreOnEpochEnd(learn_gen)

    # by default the Resnet encoder weights are frozen to Imagenet pre-trained weights,
    # keep them freeze and train only for 5 epochs to train the rest of the unet
    learn_gen.fit_one_cycle(5, callbacks=[score_callback])

    # now, unfreeze and train all layers
    learn_gen.unfreeze()

    # fit the model on 250 epochs.
    # learning rate to cycle between 1e-6 and 1e-3
    learn_gen.fit_one_cycle(250, slice(1e-6,1e-3), callbacks=[score_callback])

    learn_gen.save('unet_resnet34')

    score_callback.save('./images/training_scores.npy')

    # save a plot of the train and validation losses
    learn_gen.recorder.plot_losses()
    plt.savefig('./losses_plot.png')


def predict(learn_gen, model_weights=None, png_output_folder=None, is_computing_score=False):
    """
        Generate the increased-resolution temperature images.

        The model to generate the images will be loaded from "./images/models/unet_resnet34.pth"
        The generated images will be saved in "./images/image_gen_test.npy"

        Args:
            learn_gen (learner object): the trained model to use for prediction
            model_weights (string): filename of the weights to load
            is_computing_score (boolean): compute score for generated images if True

        Returns:
             list (boolean, float): (success True/False, score)
    """
    data_loader = Weather3to3('./images/', batch_size=4, image_size=256).data_loader
    data_loader.ignore_empty=True
    score = -1

    if model_weights is not None:
        try:
            # create the unet architecture with a Resnet-34 as encoder
            learn_gen = unet_resnet34(data_loader, weights_filename=model_weights)
        except OSError as e:
            print('  Error - model weights file do not exists: ./images/models/{}.pth'.format(model_weights))
            return (False, score)

    date_indexes = np.load('data/date_test_set.npy')
    generated_test = save_predictions(learn_gen,
                                      data_loader.valid_dl,
                                      date_indexes,
                                      png_output_folder,
                                      output_array_filename='images/image_gen_test.npy')

    if is_computing_score:
        # import score.py only here (and not at the top) because we need tensorflow install to compute score
        from score import compute_scores

        label_test = np.load('data/label_test_set.npy')
        generated_scores = compute_scores(generated_test, label_test)

        score = np.mean(generated_scores)

    return (True, score)


if __name__ == "__main__":
    """
    Arguments:
      --predict
      --train
    """
    # force the fastai library progress bar to print to the console
    master_bar, progress_bar = force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

    print('------------------------------------------')
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == '--predict':
        print('PREDICTION')
        print('  model:   ./images/models/unet_resnet34_best.pth')
        print('  dataset: ./images/valid\n')

        (success, score) = predict(learn_gen=None,
                                   model_weights='unet_resnet34_best',
                                   png_output_folder='images/valid/image_gen',
                                   is_computing_score=True)
        if success:
            print('  score =', score)
            print('  output:  ./images/valid/image_gen/   (generated increased-resolution images)')
            print('  output:  ./images/image_gen_test.npy (generated air temperature data)')

    elif arg == '--train':
        print('TRAINING THE MODEL')
        train()

    else:
        print('Usage:')
        print('  --train')
        print('  --predict')
