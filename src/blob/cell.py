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
        self._product_size = product_size

    @property
    def polygon(self):
        return self._polygon

    @property
    def vertices(self):
        return self._vertices

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
    def discretize_coordinate(coord, n_pixels):
        """Discretizes continuous coordinate in pixels
        Args:
            coord (float): continous coordinate in [0, 1]
            n_pixels (int): nb pixels in target image

        Returns:
            type: int
        """
        length = int(coord * n_pixels)
        return length

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
        print("Polygon bounds : ", polygon.bounds)
        x1, y1, x2, y2 = polygon.bounds
        width = PolygonCell.discretize_coordinate(x2 - x1, product_size[0])
        height = PolygonCell.discretize_coordinate(y2 - y1, product_size[1])
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
        h, w = product_size
        img_size = PolygonCell.img_size_from_polygon(polygon, product_size)
        print("Image size : ", img_size)

        discrete_vertices = []
        for x, y in polygon.exterior.coords:
            discrete_x = PolygonCell.discretize_coordinate(x, img_size[0])
            discrete_y = img_size[1] - PolygonCell.discretize_coordinate(y, img_size[1])
            discrete_vertices += [(discrete_x, discrete_y)]
        return discrete_vertices
