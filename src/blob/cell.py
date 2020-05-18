from PIL import Image, ImageDraw
import numpy as np
from .blob import Blob, BinaryBlob


class PolygonCell(BinaryBlob):
    """Polygon Cell blob class

    Extends/Overrides blob with:

      - Builds upon shapely.geometry.Polygon instance

    Args:
        polygon (shapely.geometry.Polygon): cell polygon
        product_size (tuple[int]): (width, height) of product the cell is
            supposed to belong to
    """
    def __init__(self, polygon, product_size, idx=None, time_serie=None,
                 sampler=None):
        img, vertices = self._shapely_to_pil(polygon, product_size)
        super().__init__(img=img, time_serie=time_serie)
        self._idx = idx
        self._size = self.img_size_from_polygon(polygon, product_size)
        self._product_size = product_size
        self._polygon = polygon
        self._vertices = vertices
        self._sampler = sampler

    def _new(self, im):
        new_im = super(Blob, self)._new(im)
        kwargs = {'polygon': self.polygon,
                  'product_size': self.product_size}
        new = self._build(**kwargs)
        super(PolygonCell, new).set_img(new_im)
        return new

    def get_center_loc(self):
        """Computes center pixel location of cell for patching to product
        Returns:
            type: tuple[int]
        """
        x1, y1, x2, y2 = self.polygon.bounds
        mean_x = 0.5 * (x1 + x2)
        mean_y = 0.5 * (y1 + y2)
        row, col = self.discretize_coordinates((mean_x, mean_y), self.product_size)
        return col, row

    def unfreeze(self):
        """Allows to iterate over blob and sets up attributes anticipating
        iteration
        """
        super().unfreeze()
        if self.sampler is not None:
            size = (self.size[1], self.size[0], self.ndim)
            self._spatial_noise = 0.5 * np.tanh(self.sampler(size=size))

    def _update_pixel_values(self, array):
        """Draws next pixel scaling vector and creates rescaled version of
            blob as array
        Returns:
            type: np.ndarray
        """
        if self.time_serie is not None:
            ts_slice = next(self._ts_iterator)
            # Scale array channel wise
            scaled_array = array * ts_slice
            if self.sampler is not None:
                # noise = self.sampler(size=scaled_array.shape)
                scaled_array += array * self._spatial_noise
            output = scaled_array.clip(min=0, max=1)
        else:
            output = self.asarray()
        return output

    def __next__(self):
        """Yields an updated version where the blob has been resized and
        its pixel values rescaled according to the specified scale sampler
        and time serie. Annotation mask is also computed and yielded along

        Returns:
            type: (np.ndarray, np.ndarray)
        """
        blob_patch = super(BinaryBlob, self).__next__()
        annotation_mask = self.annotation_mask_from(patch_array=self.asarray())
        return blob_patch, annotation_mask

    @property
    def polygon(self):
        return self._polygon

    @property
    def vertices(self):
        return self._vertices

    @property
    def size(self):
        return self._size

    @property
    def product_size(self):
        return self._product_size

    @property
    def sampler(self):
        return self._sampler

    @property
    def idx(self):
        return self._idx

    def set_idx(self, idx):
        self._idx = idx

    @staticmethod
    def _shapely_to_pil(polygon, product_size):
        """Converts shapely polygon to PIL image of dimensions scaled wrt
        product size

        Args:
        polygon (shapely.geometry.Polygon): cell polygon
        product_size (tuple[int]): (width, height) of product the cell is
            supposed to belong to

        Returns:
            type: PIL.Image.Image, list[tuple[np.ndarray]]
        """
        # Create image and compute vertices to pixel positions
        img_size = PolygonCell.img_size_from_polygon(polygon, product_size)
        img = Image.new(size=img_size, mode='L')
        vertices = PolygonCell.discretize_vertices(polygon, product_size)
        # Draw polygon on image
        draw = ImageDraw.Draw(img)
        draw.polygon(vertices, fill='white', outline='black')
        return img, vertices

    @staticmethod
    def discretize_coordinates(coord, n_pixels):
        """Discretizes continuous coordinate in pixels
        Args:
            coord (tuple[float]): (x, y) continous coordinate in [0, 1] with 0
             at lower-left corner
            n_pixels (tuple[int]): (height, width) in nb pixels for target image
             with 0 at upper-left corner

        Returns:
            type: tuple[int]
        """
        row = n_pixels[0] - int(coord[1] * n_pixels[0])
        col = int(coord[0] * n_pixels[1])
        return row, col

    @staticmethod
    def img_size_from_polygon(polygon, product_size):
        """Computes image size in pixels from shapely polygon

        Args:
            polygon (shapely.geometry.Polygon): cell polygon
            product_size (tuple[int]): (width, height) of product the cell is
                supposed to belong to
        Returns:
            type: tuple[int]
        """
        x1, y1, x2, y2 = polygon.bounds
        height, width = PolygonCell.discretize_coordinates((x2 - x1, 1 - (y2 - y1)), product_size)
        return width, height

    @staticmethod
    def discretize_vertices(polygon, product_size):
        """Computes polygon vertices coordinates in pixels given product size
            scale

        Args:
            polygon (shapely.geometry.Polygon): cell polygon
            product_size (tuple[int]): (width, height) of product the cell is
                supposed to belong to

        Returns:
            type: list[tuple[np.ndarray]]
        """
        # Compute offset position bounds
        x1, y1, x2, y2 = polygon.bounds
        bottom_left = PolygonCell.discretize_coordinates((x1, y1), product_size)
        upper_right = PolygonCell.discretize_coordinates((x2, y2), product_size)

        discrete_vertices = []
        for x, y in polygon.exterior.coords:
            discrete_x, discrete_y = PolygonCell.discretize_coordinates((x, y), product_size)
            discrete_x -= upper_right[0]
            discrete_y -= bottom_left[1]
            discrete_vertices += [(discrete_y, discrete_x)]
        return discrete_vertices
