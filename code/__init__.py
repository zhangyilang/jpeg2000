"""""""""""""""""""""""""""""""""""
JPEG2000 Image Compression script
to run: python compress.py
"""""""""""""""""""""""""""""""""""

from PIL import Image
import numpy as np
import cv2
import pywt
import math
import re

def bgr2rgb(img):   #把bgr顺序换为rgb顺序
    img=img.copy()
    temp=img[:,:,0].copy()
    img[:,:,0]=img[:,:,2].copy()
    img[:,:,2]=temp
    return img

class Tile(object):
    """
    Tile class: store img tiles generated from JPEG2000.image_tiling()
    """
    def __init__(self, tile_image):
        """
        Input: tile_image (np.array)
        Attributes: y_tile, Cb_tile, Cr_tile, original tile's y, Cb, Cr-colorspaces (np.array) generated from
        JPEG2000.component_transformation()
        """
        self.tile_image = tile_image
        self.y_tile, self.Cb_tile, self.Cr_tile = None, None, None


class JPEG2000(object):
    """compression algorithm, jpeg2000"""

    def __init__(self, file_path="data/test.jpg", quant=True, lossy=True, debug=False, tile_size=2**10):
        """
        JPEG2000 algorithm
        Initial parameters:
        file_path: path to image file to be compressed (string)
        quant: include quantization step (boolean)
        lossy: perform lossy compression (boolean)
        debug: whether to debug (boolean)
        tile_size: size of tile, default 1024 (int)
        """
        self.file_path = file_path
        self.debug = debug
        self.lossy = lossy

        # the digits of image
        self.digits = None

        # list of Tile objects of image and tile size
        self.tiles = []
        self.tile_size = tile_size

        # lossy or lossless compression component transform matrices
        if lossy:
            self.component_transformation_matrix = np.array([[0.2999, 0.587, 0.114],
                [-0.16875, -0.33126, 0.5], [0.5, -0.41869, -0.08131]])
            self.i_component_transformation_matrix = ([[1.0, 0, 1.402], [1.0, -0.34413, -0.71414], [1.0, 1.772, 0]])  
        else:
            self.component_transformation_matrix = np.array([[0.25, 0.5, 0.25],
                [0, -1.0, 1.0], [1.0, -1.0, 0]])
            self.i_component_transformation_matrix = ([[1.0, -0.25, -0.25], [1.0, -0.25, -0.75], [1.0, 0.75, -0.25]])

        # Daubechies 9/7coefficients(lossy case)
        self.dec_lo97 = [0.6029490182363579, 0.2668641184428723, -0.07822326652898785, -0.01686411844287495,
                         0.02674875741080976]
        self.dec_hi97 = [1.115087052456994, -0.5912717631142470, -0.05754352622849957, 0.09127176311424948, 0]
        self.rec_lo97 = [1.115087052456994, 0.5912717631142470, -0.05754352622849957, -0.09127176311424948, 0]
        self.rec_hi97 = [0.6029490182363579, -0.2668641184428723, -0.07822326652898785, 0.01686411844287495,
                         0.02674875741080976]

        # Le Gall 5/3 coefficients (lossless case)
        self.dec_lo53 = [6/8, 2/8, -1/8]
        self.dec_hi53 = [1, -1/2, 0]
        self.rec_lo53 = [1, 1/2, 0]
        self.rec_hi53 = [6/8, -2/8, -1/8]

        # quantization
        self.quant = quant
        self.step = 30

    def init_image(self, path):
        """ return the image at path """
        img = cv2.imread(path)
        self.digits = int(re.split(r'([0-9]+)', str(img.dtype))[1])
        return img

    def image_tiling(self, img):
        """
        tile img into square tiles based on self.tile_size (default 1024 * 1024) tiles from bottom and right edges will
        be smaller if image w and h are not divisible by self.tile_size
        """
        tile_size = self.tile_size
        (h, w, _) = img.shape  # size of original image
        counter = 0  # index number for tiles

        # change w and h to be divisible by tile_size
        left_over = w % tile_size
        w += (tile_size - left_over)
        left_over = h % tile_size
        h += (tile_size - left_over)

        # create the tiles by looping through w and h to stop on every pixel that is the top left corner of a tile
        for i in range(0, w, tile_size):  # loop through the width of img, skipping tile_size pixels every time
            for j in range(0, h, tile_size):  # loop through the height of img, skipping tile_size pixels every time
                # add the tile starting at pixel of row j and column i
                tile = Tile(img[j:j + tile_size, i:i + tile_size])
                self.tiles.append(tile)

                if self.debug:
                    cv2.imshow("tile" + str(counter), tile.tile_image)
                    cv2.imwrite("tile " + str(counter) + ".jpg", tile.tile_image)
                    counter += 1

    def dc_level_shift(self):
        # dc level shifting
        for t in self.tiles:
            #  normalization for lossy compress
            if self.lossy:
                t.tile_image = t.tile_image.astype(np.float64)
                t.tile_image -= 2 ** (self.digits - 1)
                #t.tile_image /= 2 ** self.digits
            # shift for lossless compress
            else:
                t.tile_image -= 2 ** self.digits

    def idc_level_shift(self, img):
        # inverse dc level shifting
        pass

    def component_transformation(self):
        """
        Transform every tile in self.tiles from RGB colorspace
        to either YCbCr colorspace (lossy) or YUV colorspace (lossless)
        and save the data for each color component into the tile object
        """
        # loop thorugh tiles
        for tile in self.tiles:
            (h, w, _) = tile.tile_image.shape  # size of tile

            # transform tile to RGB colorspace (library we use to view images uses BGR)
            #rgb_tile = cv2.cvtColor(tile.tile_image, cv2.COLOR_BGR2RGB)
            rgb_tile = bgr2rgb(tile.tile_image)
            Image_tile = Image.fromarray(rgb_tile, 'RGB')

            # create placeholder matrices for the different colorspace components
            # that are same w and h as original tile
            # tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.empty_like(tile.tile_image), np.empty_like(tile.tile_image), np.empty_like(tile.tile_image)
            tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
            # tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.zeros_like(tile.tile_image), np.zeros_like(tile.tile_image), np.zeros_like(tile.tile_image)

            # loop through every pixel and extract the corresponding
            # transformed colorspace values and save in tile object
            for i in range(0, w):
                for j in range(0, h):
                    r, g, b = Image_tile.getpixel((i, j))
                    rgb_array = np.array([r, g, b])
                    if self.lossy:
                        # use irreversible component transformation matrix to transform to YCbCr
                        yCbCr_array = np.matmul(self.component_transformation_matrix, rgb_array)
                    else:
                        # use reversible component transform to get YUV components
                        yCbCr_array = np.matmul(self.component_transformation_matrix, rgb_array)

                    # y = .299 * r + .587 * g + .114 * b
                    # Cb = 0
                    # Cr = 0
                    tile.y_tile[j][i], tile.Cb_tile[j][i], tile.Cr_tile[j][i] = int(yCbCr_array[0]), int(
                        yCbCr_array[1]), int(yCbCr_array[2])
                    # tile.y_tile[j][i], tile.Cb_tile[j][i], tile.Cr_tile[j][i] = int(y), int(Cb), int(Cr)

        if self.debug:
            tile = self.tiles[0]
            print(tile.y_tile.shape)
            Image.fromarray(tile.y_tile).show()

        #     # Image.fromarray(tile.y_tile).convert('RGB').save("my.jpg")

        #     # cv2.imshow("y_tile", tile.y_tile)
        #     # cv2.imshow("Cb_tile", tile.Cb_tile)
        #     # cv2.imshow("Cr_tile", tile.Cr_tile)
        #     # print tile.y_tile[0]
        #     cv2.waitKey(0)

    def i_component_transformation(self):
        """
        Inverse component transformation:
        transform all tile back to RGB colorspace
        """
        # loop through tiles, converting each back to RGB colorspace
        for tile in self.tiles:
            (h, w, _) = tile.tile_image.shape  # size of tile
            # (h, w) = tile.recovered_y_tile.shape

            # initialize recovered tile matrix to same size as original 3 dimensional tile
            tile.recovered_tile = np.empty_like(tile.tile_image)

            # loop through every pixel of the tile recovered from iDWT and use
            # the YCbCr values (if lossy) or YUV values (is lossless)
            # to transfom back to single RGB tile
            for i in range(0, w):
                for j in range(0, h):
                    y, Cb, Cr = tile.recovered_y_tile[j][i], tile.recovered_Cb_tile[j][i], tile.recovered_Cr_tile[j][i]
                    yCbCr_array = np.array([y, Cb, Cr])

                    if self.lossy:
                        # use irreversible component transform matrix to get back RGB values
                        rgb_array = np.matmul(self.i_component_transformation_matrix, yCbCr_array)
                    else:
                        # use reversible component transform to get back RGB values
                        rgb_array = np.matmul(self.i_component_transformation_matrix, yCbCr_array)
                    # save all three color dimensions to the given pixel
                    tile.recovered_tile[j][i] = rgb_array
            # break

            # if self.debug:
            #     rgb_tile = cv2.cvtColor(tile.recovered_tile, cv2.COLOR_RGB2BGR)
            #     print "rgb_tile.shape: ", rgb_tile.shape
            #     cv2.imshow("tile.recovered_tile", rgb_tile)
            #     cv2.waitKey(0)

    def dwt(self):
        """
        Run the 2-DWT (using Haar family) from the pywavelet library
        on every tile and save coefficient results in tile object
        """
        # loop through the tiles
        if self.lossy:
            wavelet = pywt.Wavelet('DB97', [self.dec_lo97, self.dec_hi97, self.rec_lo97, self.rec_hi97])
        else:
            wavelet = pywt.Wavelet('LG53', [self.dec_lo53, self.dec_hi53, self.rec_lo53, self.rec_hi53])

        for tile in self.tiles:
            # library function returns a tuple: (cA, (cH, cV, cD)), respectively LL, LH, HH, HL coefficients
            [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(tile.y_tile, wavelet, level=3)
            tile.y_coeffs = [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]
            [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(tile.Cb_tile, wavelet, level=3)
            tile.Cb_coeffs = [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]
            [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(tile.Cr_tile, wavelet, level=3)
            tile.Cr_coeffs = [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]

        if self.debug:
            names = ['cH', 'cV', 'cD']
            tile = self.tiles[0]
            for i in range(4):
                if i == 0:
                    cv2.imshow("cA3", tile.y_coeffs[i])
                else:
                    for j in range(3):
                        cv2.imshow(names[j] + str(3-i+1), tile.y_coeffs[i][j])


    def idwt(self):
        """
        Run the inverse DWT (using the Haar family) from the pywavelet library
        on every tile and save the recovered tiles in the tile object
        """
        # loop through tiles
        for tile in self.tiles:
            if self.quant:
                # if tile was quantized, need to use the recovered, un-quantized coefficients
                tile.recovered_y_tile = pywt.idwt2(tile.recovered_y_coeffs, 'haar')
                tile.recovered_Cb_tile = pywt.idwt2(tile.recovered_Cb_coeffs, 'haar')
                tile.recovered_Cr_tile = pywt.idwt2(tile.recovered_Cr_coeffs, 'haar')
            else:
                # if tile wasn't quantized, need to use the coeffs from DWT
                tile.recovered_y_tile = pywt.idwt2(tile.y_coeffs, 'haar')
                tile.recovered_Cb_tile = pywt.idwt2(tile.Cb_coeffs, 'haar')
                tile.recovered_Cr_tile = pywt.idwt2(tile.Cr_coeffs, 'haar')
                # break
        # print tile.recovered_y_tile.shape
        # print tile.recovered_Cb_tile.shape
        # print tile.recovered_Cr_tile.shape
        # tile = self.tiles[0]
        # print tile.y_tile[0]
        # print tile.recovered_y_tile[0]

    def quantization_math(self, img):
        """
        Quantize img: for every coefficient in img,
        save the original sign and decrease number of
        decimals saved by flooring the absolute value
        of the coeffcient divided by the step size
        """
        # initialize array to hold quantized coefficients,
        # to be same size as img
        (h, w) = img.shape
        quantization_img = np.empty_like(img)

        # loop through every coefficient in img
        for i in range(0, w):
            for j in range(0, h):
                # save the sign
                if img[j][i] >= 0:
                    sign = 1
                else:
                    sign = -1
                # save quantized coeffcicient
                quantization_img[j][i] = sign * math.floor(abs(img[j][i]) / self.step)
        return quantization_img

    def i_quantization_math(self, img):
        """
        Inverse quantization of img: un-quantize
        the quantized coefficients in img by
        multiplying the coeffs by the step size
        """
        # initialize array to hold un-quantized coefficients
        # to be same size as img
        (h, w) = img.shape
        i_quantization_img = np.empty_like(img)

        # loop through ever coefficient in img
        for i in range(0, w):
            for j in range(0, h):
                # save un-quantized coefficient
                i_quantization_img[j][i] = img[j][i] * self.step
        return i_quantization_img

    def quantization_helper(self, img):
        """
        Quantize the 4 different data arrays representing
        the 4 different coefficient approximations/details
        """
        cA = self.quantization_math(img[0])
        cH = self.quantization_math(img[1])
        cV = self.quantization_math(img[2])
        cD = self.quantization_math(img[3])

        return cA, cH, cV, cD

    def i_quantization_helper(self, img):
        """
        Un-quantize the 4 different data arrays representing
        the 4 different coefficient approximations/details
        """
        cA = self.i_quantization_math(img[0])
        cH = self.i_quantization_math(img[1])
        cV = self.i_quantization_math(img[2])
        cD = self.i_quantization_math(img[3])

        return cA, cH, cV, cD

    def quantization(self):
        """
        Quantize the tiles, saving the quantized
        information to the tile object
        """
        for tile in self.tiles:
            # quantize the tile in all 3 colorspaces
            tile.quantization_y = self.quantization_helper(tile.y_coeffs)
            tile.quantization_Cb = self.quantization_helper(tile.Cb_coeffs)
            tile.quantization_Cr = self.quantization_helper(tile.Cr_coeffs)

    def i_quantization(self):
        """
        Un-quantize the tiles, saving the un-quantized
        information to the tile object
        """
        for tile in self.tiles:
            tile.recovered_y_coeffs = self.i_quantization_helper(tile.quantization_y)
            tile.recovered_Cb_coeffs = self.i_quantization_helper(tile.quantization_Cb)
            tile.recovered_Cr_coeffs = self.i_quantization_helper(tile.quantization_Cr)

    def entropy_coding(self, img):
        # encode image
        pass

    def bit_stream_formation(self, img):
        # idk if we need this or what it is
        pass

    def forward(self):
        """
        Run the forward transformations to compress img
        """
        img = self.init_image(self.file_path)
        self.image_tiling(img)
        self.dc_level_shift()
        self.component_transformation()
        self.dwt()
        if self.quant:
            self.quantization()

    def backward(self):
        """
        Run the backwards transformations to get the image back
        from the compressed data
        """
        if self.quant:
            self.i_quantization()
        self.idwt()
        self.i_component_transformation()

    def run(self):
        """
        Run forward and backward transformations, saving
        compressed image data and reconstructing the image
        from the compressed data
        """
        self.forward()
        # self.backward()


if __name__ == '__main__':
    JPEG2000(debug=True).run()
