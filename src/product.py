from PIL import Image


class Product(dict):
    """Plain product class, composed of a background image and multiple blobs

    > Registers blobs as a dictionnary {idx: (location_on_bg, blob)}
    > Generates view of patched image on the fly

    Args:
        size (tuple[int]): (width, height) for background
        color (int, tuple[int]): color value for background (0-255)
        mode (str): background and blobs image mode
        blobs (dict): hand made dict formatted as {idx: (location, blob)}
    """

    def __init__(self, size, color=0, mode='L', blobs={}):
        super(Product, self).__init__(blobs)
        self.size = size
        self.bg = Image.new(size=size, color=color, mode=mode)

    def add(self, blob, loc):
        """Registers blob

        Args:
            blob (Blob): blob instance to register
            loc (tuple[int]): upper-left corner if 2-tuple, upper-left and
                lower-right corners if 4-tuple
        """
        # If blob has an id, use it
        if hasattr(blob, 'id'):
            idx = blob.id
        # Else create a new one
        else:
            idx = len(self)
        # Ensure blob is affiliated
        blob.affiliate()
        self[idx] = (loc, blob)

    def generate(self):
        """Generates image of background with patched blobs

        Returns:
            type: PIL.Image.Image
        """
        img = self.bg.copy()
        for loc, blob in self.values():
            img.paste(blob, loc)
        return img
