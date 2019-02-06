import glob

import fastai
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from fastai.vision import *
from fastai.callbacks import *


class Weather3to3(object):
    """ Weather Data Loader taking RGB png images as input and target.

        Dataset folder structure must be the following:
            ├── images
            │   ├── train
            │   │   ├── hires
            │   │   │   ├── 2016100100.png
            │   │   │   └── 2016100103.png
            │   │   └── lowres
            │   │   │   ├── 2016100100.png
            │   │   │   └── 2016100103.png
            │   └── valid
            │       ├── hires
            │       │   ├── 2016100100.png
            │       │   └── 2016100103.png
            │       └── lowres
            │           ├── 2016100100.png
            │           └── 2016100103.png
    """

    def __init__(self, image_path, batch_size=16, image_size=256):
        """
        Create a Fastai ImageDataBunch object from given path.

        Args:
            image_path (string): path leading to train and valid folder as described above in this Class documentation
            batch_size (integer): data loader batch size
            image_size (integer): image size of input and output images
        """

        # Data augmentation: flip horizontal and vertical as well as 90 degree rotation
        xform = get_transforms(
            do_flip = True,
            flip_vert = True,
            max_rotate = None,
            max_zoom = 1,
            max_lighting = None,
            max_warp = None
        )

        # Create the Fastai DataBunch using the data block api:
        # https://docs.fast.ai/data_block.html
        self.data_loader = (
            ImageImageList.from_folder(image_path)
                # filter to have only /lowres/ as source
                .filter_by_func(lambda path: 'lowres' in path.parent.as_posix())

                # use the folders to split in train/valid
                .split_by_folder()

                # change /lowres/ to /hires/ to match source with target
                .label_from_func(lambda path: path.parents[1]/'hires'/path.name)

                # apply the data augmentation transforms to both x and y images (lowres and hires)
                .transform(xform, size=image_size, tfm_y=True)

                .databunch(bs=batch_size)

                # convert the LabelLists in ImageDataBunch and normalize to imagenet both X and Y
                .normalize(imagenet_stats, do_x=True, do_y=True)
        )

        # the unet_learner need a variable c to be the number of classes
        self.data_loader.c = 3


class Weather8to1(object):
    """ Weather Data Loader taking 8 channels 256x256 images as input and 1 channel 256x256 image as target.

        Each image is to be stored in .npy file:
            array of numpy array (256x256x8 for input and 256x256x1 for target)
    """

    def __init__(self,
                 train_x_folder, train_y_folder,
                 valid_x_folder, valid_y_folder,
                 batch_size=16,
                 num_workers=4):
        """
        Create a Fastai ImageDataBunch object from given paths.
        Each filename in x_folder must have a correspoding filename in y_folder.

        Args:
            train_x_folder (string): path leading to input train folder containing the .npy files
            train_y_folder (string): path leading to target train folder containing the .npy files
            valid_x_folder (string): path leading to input valid folder containing the .npy files
            valid_y_folder (string): path leading to target valid folder containing the .npy files

            batch_size (integer): data loader batch size
            image_size (integer): image size of input and output images
        """
        train_dataset = Dataset8to1(train_x_folder, train_y_folder, is_valid_set=False)
        valid_dataset = Dataset8to1(valid_x_folder, valid_y_folder, is_valid_set=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.data_loader = ImageDataBunch(train_dl=train_dataloader, valid_dl=valid_dataloader)
        self.data_loader.c = 1

class Dataset8to1(Dataset):
    """
    Create a Dataset object from given paths.

    Each image is to be stored in .npy file:
        array of numpy array (256x256x8 for input and 256x256x1 for target)

    Data augmentation is applied to all images (from x_folder and y_folder):
        - flip horizontally
        - flip vertically
        - rotation 90 and 270 degree
    """

    def __init__(self, x_folder, y_folder, is_valid_set=False):
        """
        Create a Dataset object from given paths.

        Args:
            x_folder (String): Complete pathname to x sample.
            y_folder (String): Complete pathname to y sample.
            is_valid_set (boolean): True for validation set to avoid applying the transform
        """
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.is_valid_set = is_valid_set

        # get the list of .npy in the x_folder (complete filename and path in string format)
        self.x_filenames = sorted(glob.glob(os.path.join(x_folder, '*.npy')))

        self.transform_no_flip = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_h_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomHorizontalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_v_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomVerticalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_hv_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomHorizontalFlip(flip=True),
            CustomVerticalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_r90 = transforms.Compose([
            transforms.ToTensor(),
            CustomRotation(angle=90),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_r270 = transforms.Compose([
            transforms.ToTensor(),
            CustomRotation(angle=270),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.x_filenames)

    def _get_random_transform(self):
        # we do not apply any data augmentation transform on valid/test set
        if self.is_valid_set:
            return self.transform_no_flip

        r = random.randrange(6)

        # r == 0
        xform = self.transform_no_flip

        if r == 1:
            xform = self.transform_hv_flip
        elif r == 2:
            xform = self.transform_v_flip
        elif r == 3:
            xform = self.transform_h_flip
        elif r == 4:
            xform = self.transform_r90
        elif r == 5:
            xform = self.transform_r270

        return xform

    def __getitem__(self, idx):
        # extract filename (without path)
        filename = Path(self.x_filenames[idx]).name

        x_npy_pathname = os.path.join(self.x_folder, filename)
        y_npy_pathname = os.path.join(self.y_folder, filename)

        # numpy.ndarray (H x W x C) in the range [0, 1]
        x_image = np.load(x_npy_pathname)
        y_image = np.load(y_npy_pathname)

        xform = self._get_random_transform()

        # ToTensor() in the transform will convert the pil image to
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        x_image = xform(x_image)
        y_image = xform(y_image)

        #x_image = x_image[0:5,:,:]

        return x_image, y_image

    def __repr__(self):
        item0 = self.__getitem__(0)

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    lowres images: {}\n'.format(self.x_folder)
        fmt_str += '    hires images: {}\n'.format(self.y_folder)

        fmt_str += '    x image shape = {}\n'.format(item0[0].shape)
        fmt_str += '    Y image shape = {}\n'.format(item0[1].shape)
        fmt_str += '    is_valid_set = {}\n'.format(self.is_valid_set)

        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform_no_flip.__repr__().
                                     replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class Weather5to1(object):
    """ Weather Data Loader taking 8 channels 256x256 images as input and 1 channel 256x256 image as target.

        Each image is to be stored in .npy file:
            array of numpy array (256x256x8 for input and 256x256x1 for target)
    """

    def __init__(self,
                 train_x_folder, train_y_folder,
                 valid_x_folder, valid_y_folder,
                 batch_size=16,
                 num_workers=4):
        """
        Create a Fastai ImageDataBunch object from given paths.
        Each filename in x_folder must have a correspoding filename in y_folder.

        Args:
            train_x_folder (string): path leading to input train folder containing the .npy files
            train_y_folder (string): path leading to target train folder containing the .npy files
            valid_x_folder (string): path leading to input valid folder containing the .npy files
            valid_y_folder (string): path leading to target valid folder containing the .npy files

            batch_size (integer): data loader batch size
            image_size (integer): image size of input and output images
        """
        train_dataset = Dataset5to1(train_x_folder, train_y_folder, is_valid_set=False)
        valid_dataset = Dataset5to1(valid_x_folder, valid_y_folder, is_valid_set=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.data_loader = ImageDataBunch(train_dl=train_dataloader, valid_dl=valid_dataloader)
        self.data_loader.c = 1

class Dataset5to1(Dataset):
    """
    Create a Dataset object from given paths.

    Each image is to be stored in .npy file:
        array of numpy array (256x256x8 for input and 256x256x1 for target)

    Data augmentation is applied to all images (from x_folder and y_folder):
        - flip horizontally
        - flip vertically
        - rotation 90 and 270 degree
    """

    def __init__(self, x_folder, y_folder, is_valid_set=False):
        """
        Create a Dataset object from given paths.

        Args:
            x_folder (String): Complete pathname to x sample.
            y_folder (String): Complete pathname to y sample.
            is_valid_set (boolean): True for validation set to avoid applying the transform
        """
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.is_valid_set = is_valid_set

        # get the list of .npy in the x_folder (complete filename and path in string format)
        self.x_filenames = sorted(glob.glob(os.path.join(x_folder, '*.npy')))

        self.transform_no_flip = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_h_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomHorizontalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_v_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomVerticalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_hv_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomHorizontalFlip(flip=True),
            CustomVerticalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_r90 = transforms.Compose([
            transforms.ToTensor(),
            CustomRotation(angle=90),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_r270 = transforms.Compose([
            transforms.ToTensor(),
            CustomRotation(angle=270),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.x_filenames)

    def _get_random_transform(self):
        # we do not apply any data augmentation transform on valid/test set
        if self.is_valid_set:
            return self.transform_no_flip

        r = random.randrange(6)

        # r == 0
        xform = self.transform_no_flip

        if r == 1:
            xform = self.transform_hv_flip
        elif r == 2:
            xform = self.transform_v_flip
        elif r == 3:
            xform = self.transform_h_flip
        elif r == 4:
            xform = self.transform_r90
        elif r == 5:
            xform = self.transform_r270

        return xform

    def __getitem__(self, idx):
        # extract filename (without path)
        filename = Path(self.x_filenames[idx]).name

        x_npy_pathname = os.path.join(self.x_folder, filename)
        y_npy_pathname = os.path.join(self.y_folder, filename)

        # numpy.ndarray (H x W x C) in the range [0, 1]
        x_image = np.load(x_npy_pathname)
        y_image = np.load(y_npy_pathname)

        xform = self._get_random_transform()

        # ToTensor() in the transform will convert the pil image to
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        x_image = xform(x_image)
        y_image = xform(y_image)

        x_image = x_image[0:5,:,:]

        return x_image, y_image

    def __repr__(self):
        item0 = self.__getitem__(0)

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    lowres images: {}\n'.format(self.x_folder)
        fmt_str += '    hires images: {}\n'.format(self.y_folder)

        fmt_str += '    x image shape = {}\n'.format(item0[0].shape)
        fmt_str += '    Y image shape = {}\n'.format(item0[1].shape)
        fmt_str += '    is_valid_set = {}\n'.format(self.is_valid_set)

        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform_no_flip.__repr__().
                                     replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class Weather5to5(object):
    """ Weather Data Loader taking 5 channels 256x256 images as input and 5 channel 256x256 image as target.

        Each image is to be stored in .npy file:
            array of numpy array (256x256x5 for input and 256x256x5 for target)
    """

    def __init__(self,
                 train_x_folder, train_y_folder,
                 valid_x_folder, valid_y_folder,
                 batch_size=16,
                 num_workers=4):
        """
        Create a Fastai ImageDataBunch object from given paths.
        Each filename in x_folder must have a correspoding filename in y_folder.

        Args:
            train_x_folder (string): path leading to input train folder containing the .npy files
            train_y_folder (string): path leading to target train folder containing the .npy files
            valid_x_folder (string): path leading to input valid folder containing the .npy files
            valid_y_folder (string): path leading to target valid folder containing the .npy files

            batch_size (integer): data loader batch size
            image_size (integer): image size of input and output images
        """
        train_dataset = Dataset5to5(train_x_folder, train_y_folder, is_valid_set=False)
        valid_dataset = Dataset5to5(valid_x_folder, valid_y_folder, is_valid_set=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.data_loader = ImageDataBunch(train_dl=train_dataloader, valid_dl=valid_dataloader)
        self.data_loader.c = 5

class Dataset5to5(Dataset):
    """
    Create a Dataset object from given paths.

    Each image is to be stored in .npy file:
        array of numpy array (256x256x5 for input and 256x256x5 for target)

    Data augmentation is applied to all images (from x_folder and y_folder):
        - flip horizontally
        - flip vertically
        - rotation 90 and 270 degree
    """

    def __init__(self, x_folder, y_folder, is_valid_set=False):
        """
        Create a Dataset object from given paths.

        Args:
            x_folder (String): Complete pathname to x sample.
            y_folder (String): Complete pathname to y sample.
            is_valid_set (boolean): True for validation set to avoid applying the transform
        """
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.is_valid_set = is_valid_set

        # get the list of .npy in the x_folder (complete filename and path in string format)
        self.x_filenames = sorted(glob.glob(os.path.join(x_folder, '*.npy')))

        self.transform_no_flip = transforms.Compose([
            transforms.ToTensor(),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_h_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomHorizontalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_v_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomVerticalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_hv_flip = transforms.Compose([
            transforms.ToTensor(),
            CustomHorizontalFlip(flip=True),
            CustomVerticalFlip(flip=True),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_r90 = transforms.Compose([
            transforms.ToTensor(),
            CustomRotation(angle=90),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

        self.transform_r270 = transforms.Compose([
            transforms.ToTensor(),
            CustomRotation(angle=270),

            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.x_filenames)

    def _get_random_transform(self):
        # we do not apply any data augmentation transform on valid/test set
        if self.is_valid_set:
            return self.transform_no_flip

        r = random.randrange(6)

        # r == 0
        xform = self.transform_no_flip

        if r == 1:
            xform = self.transform_hv_flip
        elif r == 2:
            xform = self.transform_v_flip
        elif r == 3:
            xform = self.transform_h_flip
        elif r == 4:
            xform = self.transform_r90
        elif r == 5:
            xform = self.transform_r270

        return xform

    def __getitem__(self, idx):
        # extract filename (without path)
        filename = Path(self.x_filenames[idx]).name

        x_npy_pathname = os.path.join(self.x_folder, filename)
        y_npy_pathname = os.path.join(self.y_folder, filename)

        # numpy.ndarray (H x W x C) in the range [0, 1]
        x_image = np.load(x_npy_pathname)
        y_image = np.load(y_npy_pathname)

        xform = self._get_random_transform()

        # ToTensor() in the transform will convert the pil image to
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        x_image = xform(x_image)
        y_temp_image = xform(y_image)

        # keep only the 5 first channels out of 8 from the .npy files
        x_image = x_image[0:5,:,:]

        # y is a copy of x but with its own temperature channel
        y_image = x_image[0:5,:,:].clone()
        y_image[0,:,:] = y_temp_image[0,:,:]

        return x_image, y_image

    def __repr__(self):
        item0 = self.__getitem__(0)

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    lowres images: {}\n'.format(self.x_folder)
        fmt_str += '    hires images: {}\n'.format(self.y_folder)

        fmt_str += '    x image shape = {}\n'.format(item0[0].shape)
        fmt_str += '    Y image shape = {}\n'.format(item0[1].shape)
        fmt_str += '    is_valid_set = {}\n'.format(self.is_valid_set)

        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform_no_flip.__repr__().
                                     replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CustomHorizontalFlip(object):
    """Horizontally flip the given tensor array Image (C x H x W).

    Args:
        flip (boolean): flip the image if True
    """
    def __init__(self, flip=True):
        self.flip = flip

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (tensor array (C x H x W): Image to be flipped.

        Returns:
            (tensor array (C x H x W): original or flipped image based on flip param
        """
        if self.flip:
            return img_tensor.flip(2)
        return img_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(flip={})'.format(self.flip)


class CustomVerticalFlip(object):
    """Vertically flip the given tensor array Image (C x H x W).

    Args:
        flip (boolean): flip the image if True
    """
    def __init__(self, flip=True):
        self.flip = flip

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (tensor array (C x H x W): Image to be flipped.

        Returns:
            (tensor array (C x H x W): original or flipped image based on flip param
        """
        if self.flip:
            return img_tensor.flip(1)
        return img_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(flip={})'.format(self.flip)


class CustomRotation(object):
    """Rotate the given tensor array Image (C x H x W).

    Args:
        angle (integer): Rotatio angle in degree: 0, 90 or 270
    """
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (tensor array (C x H x W): Image to be rotated.

        Returns:
            (tensor array (C x H x W): original or rotated image based on angle param
        """
        if self.angle == 90:
            return img_tensor.transpose(1,2).flip(2)

        if self.angle == 270:
            return img_tensor.transpose(1,2).flip(1)

        return img_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(angle={})'.format(self.angle)