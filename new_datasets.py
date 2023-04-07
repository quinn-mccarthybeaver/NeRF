import json
import numpy as np
import imageio.v3 as iio
from skimage.transform import resize
#from PIL import Image


class data_preprocessing:

    def __init__(self, data_path='', mode='', target_size=(800, 800)):
        self.data_path = data_path
        self.mode = mode
        self.json_name = f'transforms_{mode}.json'
        self.img_wh = target_size
        self.images = []
        self.c2ws = []
        self.N = 0

        self.jsonPath = f'{self.data_path}/{self.json_name}'
        self.image_path = f'{self.data_path}/{mode}'

        with open(self.jsonPath, 'r') as file:
            data = json.load(file)

        self.fieldOfView = data["camera_angle_x"]
        self.Frames = data["frames"]

        self.focal = 0.5 * 800 / np.tan(0.5 * self.fieldOfView)  # original focal length

        # when W=800

        self.focal = self.focal * self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        print("focal_length: ", self.focal)

        """self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w / 2
        self.K[1, 2] = h / 2
        print("K-MATRIX-INRIX", self.K)"""

        for frame in self.Frames:
            imagePath = frame["file_path"]
            imagePath = imagePath.replace(".", self.data_path)
            imagePath = f"{imagePath}.png"
            # read image
            img = iio.imread(imagePath) / 255.
            # img_path = self.data_path + '/imgs/' + name.replace('txt', 'png')
            # img = Image.open(img_path)
            # img = np.array(img) / 255.0
            # resize image
            img = resize(img, self.img_wh)

            # convert and append images in 4D  (B, W, H, C) # B =1
            self.images.append(img[None, ...])

            c2w = frame["transform_matrix"]
            self.c2ws.append(c2w)

        # concatenate ALL the images (B, W, H, C) B = total number of images
        self.images = np.concatenate(self.images)
        self.c2ws = np.array(self.c2ws)

        print(self.c2ws.shape)
        print(self.c2ws[27])
        print(type(self.c2ws), type(self.c2ws[27]))

        self.N = self.images.shape[0]

        print("size: ", self.images.shape)

        # convert the images from RGBA TO RGB (RGBA -> RGB)
        if self.images.shape[3] == 4:
            self.images = self.images[..., :3] * self.images[..., -1:] + (1 - self.images[..., -1:])
        print("size: ", self.images.shape)

    def get_rays(self):

        print(self.N, self.img_wh[1] * self.img_wh[0])

        rays_o = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
        rays_d = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
        target_px_values = self.images.reshape((self.N, self.img_wh[1] * self.img_wh[0], 3))

        for i in range(self.N):
            c2w = self.c2ws[i]
            f = self.focal

            u = np.arange(self.img_wh[0])
            v = np.arange(self.img_wh[1])
            u, v = np.meshgrid(u, v)
            dirs = np.stack((u - self.img_wh[0] / 2, -(v - self.img_wh[1] / 2), - np.ones_like(u) * f), axis=-1)
            dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

            rays_d[i] = dirs.reshape(-1, 3)
            rays_o[i] += c2w[:3, -1]

        print("######################################################################")
        print('ray origin is:',rays_o.shape)
        print('ray direction is:',rays_d.shape)
        print('target pixel values is:',target_px_values.shape)

        return rays_o, rays_d, target_px_values, self.N
