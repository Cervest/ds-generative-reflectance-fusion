from PIL import Image, ImageDraw
from .blob import Blob


class PolygonCell(Blob):
    """Polygon Cell blob class

    Extends/Overrides blob with:

      - Builds upon shapely.geometry.Polygon instance
    Args:
        polygon (shapely.geometry.Polygon): cell polygon
        product_size (tuple[int]): (width, height) of product the cell is
            supposed to belong to
    """
    def __init__(self, polygon, product_size):
        img, vertices = self._shapely_to_pil(polygon, product_size)
        super().__init__(img=img)
        self._polygon = polygon
        self._vertices = vertices
        self._size = self.img_size_from_polygon(polygon, product_size)
        self._product_size = product_size

    def _new(self, im):
        new_im = super(Blob, self)._new(im)
        kwargs = {'polygon': self.polygon,
                  'product_size': self.product_size}
        new = self._build(**kwargs)
        super(PolygonCell, new).set_img(new_im)
        return new

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
        width, height = PolygonCell.discretize_coordinates((x2 - x1, 1 - (y2 - y1)), product_size)
        return height, width

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
