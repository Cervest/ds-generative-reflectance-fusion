from skimage.measure import block_reduce


class Degrade:
    """

    Args:
        size (tuple[int]): target (width, height) fo degraded product
        transform (callable): geometric transformation to apply
        temporal_res (int): temporal resolution of degraded product
        aggregate_fn (type): aggregation function used for downsampling
    """

    def __init__(self, size, transform=None, temporal_res=None, aggregate_fn=None):
        self._size = size
        self._transform = transform
        self._temporal_res = temporal_res
        self._aggregate_fn = aggregate_fn

    def apply_transform(self, img, seed=None):
        if self.transform:
            output = self.transform(img, seed=seed)
        else:
            output = img
        return output

    def downsample(self, img):
        # Compute aggregation blocks dimensions
        width, height, _ = img.shape
        block_width = width // self.size[0]
        block_height = height // self.size[1]
        block_size = (block_width, block_height, 1)
        # Apply downsampling
        down_img = block_reduce(image=img, block_size=block_size, func=self.aggregate_fn)
        return down_img

    def __call__(self, img, seed=None):
        img = self.apply_transform(img, seed=seed)
        img = self.downsample(img)
        return img

    @property
    def size(self):
        return self._size

    @property
    def transform(self):
        return self._transform

    @property
    def temporal_res(self):
        return self._temporal_res

    @property
    def aggregate_fn(self):
        return self._aggregate_fn
