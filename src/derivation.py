import numpy as np
from skimage.measure import block_reduce
from progress.bar import Bar
from .product import ProductExport
from src.utils import setseed


class Degrader:
    """
    Degradation class for toy generated imagery products

        > Applies geometric transformation to images (typically perspective)
        > Downsamples by blocks aggregation

    Currently, blocks are designed to not intersect such that only fractions of input
    image size can be set as target size. To be solved adding padding or stride.

    Args:
        size (tuple[int]): target (width, height) fo degraded product
        transform (callable): geometric transformation to apply
        temporal_res (int): temporal resolution of degraded product
        aggregate_fn (type): aggregation function used for downsampling
    """
    def __init__(self, size, transform=None, temporal_res=1, aggregate_fn=np.mean):
        self._size = size
        self._transform = transform
        self._temporal_res = temporal_res
        self._aggregate_fn = aggregate_fn

    def _new_index_from(self, index):
        new_index = index.copy()
        new_index['features']['width'] = self.size[0]
        new_index['features']['height'] = self.size[1]
        new_index['features']['nframes'] = 0
        new_index['files'] = {}
        return new_index

    @setseed('numpy')
    def apply_transform(self, img, seed=None):
        """Applies image transformation on numpy array
        Args:
            img (np.ndarray)
            seed (int): random seed
        Returns:
            type: np.ndarray
        """
        if self.transform:
            output = self.transform(image=img)
        else:
            output = img
        return output

    def downsample(self, img):
        """Runs image downsampling with custom aggregation function
        Args:
            img (np.ndarray)
        Returns:
            type: np.ndarray
        """
        # Compute aggregation blocks dimensions
        width, height, _ = img.shape
        block_width = width // self.size[0]
        block_height = height // self.size[1]
        block_size = (block_width, block_height, 1)
        # Apply downsampling
        down_img = block_reduce(image=img, block_size=block_size, func=self.aggregate_fn)
        return down_img

    def __call__(self, img, seed=None):
        """Applies class defined image alteration successively :
            - Geometric transformation of image
            - Image downsampling to specified spatial resolution
        Args:
            img (np.ndarray)
            seed (int): random seed
        Returns:
            type: np.ndarray
        """
        img = self.apply_transform(img, seed=seed)
        img = self.downsample(img)
        return img

    def derive(self, product_set, output_dir):
        """Iterates over product dataset, applies degradation transformation
            and dumps resulting images

        Args:
            product_set (ProductDataset): instance of product dataset typically
                previously generate with Product class
            output_dir (str): path to output directory
        """
        export = ProductExport(output_dir, astype='h5')
        export._setup_output_dir()
        bar = Bar("Derivation", max=len(product_set))
        # Build new index from dataset's one
        index = self._new_index_from(product_set.index)
        export.set_index(index)

        for i in range(len(product_set)):
            # Retrieve image from dataset
            img, _ = product_set[i]
            if i % self.temporal_res == 0:
                # If step matches temporal resolution, degrade and dump image
                img = self(img)
                index['features']['nframes'] += 1

                frame_name = f"frame_{i}.h5"
                annotation_name = f"annotation_{i}.h5"
                export.add_to_index(i, frame_name, annotation_name)
                export.dump_frame(img, frame_name)
                # export.dump_annotation(., annotation_name)
            else:
                # Else skip image
                index['files'][i] = None
            bar.next()
        export.dump_index(index=index)

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
