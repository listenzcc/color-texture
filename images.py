# %%
import cv2

from pathlib import Path

import matplotlib.pyplot as plt

# %%
folder = Path('assets/image')

file_names = [e for e in folder.iterdir()]
file_names


# %%

# The default_position controls ROI of the picture,
# the picture will be cut to fit the rect.
# The format is the 4x rect, (x, y, w, h)

default_position = (0, 0, 10000, 10000)
# default_position = (200, 580, 700, 700)


class Image(object):
    def __init__(self, path, position=default_position):
        self.path = path
        self.position = position
        self.read_img()
        pass

    def read_img(self):
        path = self.path
        if isinstance(path, Path):
            path = path.as_posix()

        position = self.position

        img_color = cv2.imread(path)

        # img_color = img_color[position[0]:position[0] + position[2],
        #                       position[1]:position[1] + position[3]]
        img_color = self._patch(img_color, position)

        img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        img_hls = cv2.cvtColor(img_color, cv2.COLOR_BGR2HLS)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        self.color = img_color
        self.hls = img_hls
        self.rgb = img_rgb
        self.gray = img_gray
        self.shape = img_color.shape

    def _patch(self, mat, position):
        mat = mat[position[0]:position[0] + position[2],
                  position[1]:position[1] + position[3]]
        return mat

    def get(self, position):
        return [
            self._patch(e, position)
            for e in [self.color, self.hls, self.rgb, self.gray]
        ]

    def plot(self):
        # ! The figsize uses the unit of inches
        # ! Do not mix it with pixel, it will make the figure TOO LARGE
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

        axe = axes[0]
        axe.imshow(self.rgb)

        # axe = axes[1][0]
        # axe.imshow(self.hls)

        plt.style.use('grayscale')
        axe = axes[1]
        axe.imshow(self.gray)


# %%

if __name__ == '__main__':
    image = Image(file_names[0])

    image.plot()
    plt.tight_layout()
    plt.show()


# %%
