import numpy as np
from skimage.measure import block_reduce
from progress.bar import Bar
from .product import ProductExport
from src.utils import setseed


class Degrader:
    """
    Degradation class for toy generated imagery products

        > Applies geometric and corruption transformation to images (typically perspective)
        > Downsamples by blocks aggregation

    Currently, blocks are designed to not intersect such that only fractions of input
    image size can be set as target size. To be solved adding padding or stride.

    Args:
        size (tuple[int]): target (width, height) fo degraded product
        corruption_transform (callable): image corruption transform to apply
        geometric_transform (callable): geometric transformation to apply
        postprocess_transform (callable): postprocessing transformation to apply
            at very last step
        temporal_res (int): temporal resolution of degraded product in days
        aggregate_fn (type): aggregation function used for downsampling
    """
    def __init__(self, size, temporal_res=1, corruption_transform=None,
                 geometric_transform=None, postprocess_transform=None,
                 aggregate_fn=np.mean):
        self._size = size
        self._corruption_transform = corruption_transform
        self._geometric_transform = geometric_transform
        self._postprocess_transform = postprocess_transform
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
    def apply_corruption_transform(self, img, seed=None):
        """Applies image corruption transformation on numpy array
        Args:
            annotation (np.ndarray)
            seed (int): random seed
        Returns:
            type: np.ndarray
        """
        if self.corruption_transform:
            img = self.corruption_transform(image=img)
        return img

    def apply_geometric_transform(self, img):
        """Applies image geometric transformation on numpy array
        Args:
            annotation (np.ndarray)
        Returns:
            type: np.ndarray
        """
        if self.geometric_transform:
            img = self.geometric_transform(image=img)
        return img

    @setseed('numpy')
    def apply_postprocess_transform(self, img, seed=None):
        """Applies image postprocessing transformation on numpy array
        Args:
            annotation (np.ndarray)
        Returns:
            type: np.ndarray
        """
        if self.postprocess_transform:
            img = self.postprocess_transform(image=img)
        return img

    def apply_transform(self, img, seed=None):
        """Applies image corruption and geometric transformation on numpy array
        Args:
            img (np.ndarray)
            seed (int): random seed
        Returns:
            type: np.ndarray
        """
        img = self.apply_corruption_transform(img=img, seed=seed)
        img = self.apply_geometric_transform(img=img)
        return img

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

    def transform_annotation(self, annotation):
        """Applies geometric and downsampling transforms to annotation mask
            as we don't want mask corruption
        Args:
            annotation (np.ndarray): annotation mask
        Returns:
            type: np.ndarray
        """
        annotation = self.apply_geometric_transform(img=annotation)
        annotation = self.downsample(img=annotation)
        return annotation

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
        img = self.apply_postprocess_transform(img, seed=seed)
        return img

    def derive(self, product_set, output_dir):
        """Iterates over product dataset, applies degradation transformation
            and dumps resulting images

        Args:
            product_set (ProductDataset): instance of product dataset typically
                previously generate with Product class
            output_dir (str): path to output directory
        """
        # Setup export
        export = ProductExport(output_dir, astype='h5')
        export._setup_output_dir()
        bar = Bar("Derivation", max=len(product_set))

        # Build new index from dataset's one
        index = self._new_index_from(product_set.index)
        export.set_index(index)

        for i in range(len(product_set)):
            # Retrieve image from dataset
            img, annotation = product_set[i]

            # If step matches temporal resolution
            if i % self.temporal_res == 0:
                # Degrade image and annotation
                img = self(img=img)
                annotation = self.transform_annotation(annotation=annotation)

                # Record new frame in index
                frame_name = f"frame_{i}.h5"
                annotation_name = f"annotation_{i}.h5"
                index['features']['nframes'] += 1
                export.add_to_index(i, frame_name, annotation_name)

                # Dump degraded frame and annotation mask
                export.dump_frame(img, frame_name)
                export.dump_annotation(annotation, annotation_name)
            else:
                # Else skip image
                index['files'][i] = None
            bar.next()
        export.dump_index(index=index)

    @property
    def size(self):
        return self._size

    @property
    def corruption_transform(self):
        return self._corruption_transform

    @property
    def geometric_transform(self):
        return self._geometric_transform

    @property
    def postprocess_transform(self):
        return self._postprocess_transform

    @property
    def temporal_res(self):
        return self._temporal_res

    @property
    def aggregate_fn(self):
        return self._aggregate_fn
