import numpy as np
import random
import torch
import torchvision.transforms.functional as F

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
    This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, labels = sample['image'], sample['labels']
        image = normalize(image, self.mean, self.std, self.inplace)
        return {'image': image, 'labels': labels}


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
    This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = {}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=torch.uint8)
    std = torch.as_tensor(std, dtype=torch.uint8)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


class Resize(object):
    """Rescale the image in a sample to a given size.

    Args:
    output_size (tuple or int): Desired output size. If tuple, output is
    matched to output_size. If int, smaller of image edges is matched
    to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        np_image = np.array(image)
        h, w = np_image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = F.resize(image, (new_h, new_w))
        labels = F.resize(labels, (new_h, new_w))
        return {'image': image, 'labels': labels}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
        img (PIL Image): Image to be flipped.

        Returns:
        PIL Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']

        if random.random() < self.p:
            image = F.hflip(image)
            labels = F.hflip(labels)
        return {'image': image, 'labels': labels} 

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']

        if random.random() < self.p:
            image = F.vflip(image)
            labels = F.vflip(labels)
        return {'image': image, 'labels': labels}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        temp_image = np.array(image)
        temp_image = temp_image.transpose(2, 0, 1)
        temp_labels = np.array(labels)
        pattern = np.array([0, 0, 0])
        mask = (temp_labels == pattern).all(axis=2)
        temp_labels = np.where(mask, 0, 1) # Black [0, 0, 0] labeled as 0; white as 1
        return {'image': torch.from_numpy(temp_image), 'labels': torch.from_numpy(temp_labels)}
