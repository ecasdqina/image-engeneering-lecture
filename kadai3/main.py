from argparse import ArgumentParser
from typing import NoReturn, List, Tuple
from PIL import Image
from math import log10
from itertools import product

import numpy as np


# DCT
# ref: https://tony-mooori.blogspot.com/2016/02/dctpythonpython.html

class DCT:
    def __init__(self):
        self.N = 8
        self.phi_1d = np.array([self.phi(i) for i in range(self.N)])
        self.phi_2d = np.zeros((self.N, self.N, self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i * phi_j

    def dct(self, data):
        return self.phi_1d.dot(data)

    def idct(self, c):
        return np.sum(self.phi_1d.T * c, axis=1)

    def dct2(self, data):
        return np.sum(self.phi_2d.reshape(self.N * self.N, self.N * self.N) *
                      data.reshape(self.N * self.N), axis=1).reshape(self.N, self.N)

    def idct2(self, c):
        return np.sum((c.reshape(self.N, self.N, 1) * self.phi_2d.reshape(self.N, self.N, self.N * self.N))
                      .reshape(self.N * self.N, self.N * self.N), axis=0).reshape(self.N, self.N)

    def phi(self, k):
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        else:
            return np.sqrt(2.0 / self.N) * np.cos((k * np.pi / (2 * self.N)) * (np.arange(self.N) * 2 + 1))


transformer = DCT()


def expand_image(image: Image):
    """画像の大きさを 8 の倍数にする"""

    array = np.array(image)

    while array.shape[0] % 8 != 0:
        array = np.insert(array, array.shape[0], [0] * array.shape[1], axis=0)
    while array.shape[1] % 8 != 0:
        array = np.insert(array, array.shape[1], [0] * array.shape[0], axis=1)
    assert (array.shape[0] % 8, array.shape[1] % 8) == (0, 0)
    return Image.fromarray(array)


def dct(image: Image, method, args) -> np.array:
    array, size = np.array(image), image.size

    d = np.zeros(size)
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            sep = array[i:i + 8, j:j + 8]

            tmp = transformer.dct2(sep)
            tmp = method(tmp, args)
            d[i:i + 8, j:j + 8] = tmp
    return d


def idct(array: np.array) -> Image:
    d, size = array.copy(), array.shape
    for i in range(0, size[0], 8):
        for j in range(0, size[1], 8):
            sep = array[i:i + 8, j:j + 8]

            d[i:i + 8, j:j + 8] = transformer.idct2(sep)
    return Image.fromarray(np.round(d).astype(np.uint8))


def method_1(dct: np.array, args) -> np.array:
    """左下三角部分を 0 に"""

    dct = dct.copy()
    for i in range(1, dct.shape[0]):
        dct[i][-i:] = 0
    return dct


def method_2(dct: np.array, args) -> np.array:
    """閾値 t 以下を 0 に"""

    return np.where((-args.threshold <= dct) & (dct <= args.threshold), 0, dct)


def method_3(dct: np.array, args) -> np.array:
    """下位 s% を 0 に"""

    t = np.sort(np.abs(dct.flatten()))[dct.size * args.small // 100]
    return np.where((-t <= dct) & (dct <= t), 0, dct)


def zigzag_scan() -> List[Tuple[int, int]]:
    x, y, dx, dy, scan = 0, 0, -1, 1, []
    while True:
        scan.append((x, y))

        if x == 7 and y == 0:
            break

        x += dx
        y += dy
        if y == -1 or x == -1:
            x += 1 if x == -1 else 0
            y += 1 if y == -1 else 0

            dx *= -1
            dy *= -1

    x, y, dx, dy = 7, 1, -1, 1
    while True:
        scan.append((x, y))

        if x == 7 and y == 7:
            break

        x += dx
        y += dy
        if y == 8 or x == 8:
            x += -1 if x == 8 else 2
            y += -1 if y == 8 else 2

            dx = 1 if dx == -1 else -1
            dy = -1 if dy == 1 else 1
    return scan


def method_4(dct: np.array, args) -> np.array:
    """ジグザグスキャン末尾 s% を 0 に"""

    scan = zigzag_scan()[::-1][:dct.size * args.small // 100]

    d = dct.copy()
    for x, y in scan:
        d[x][y] = 0
    return d


def method_5(dct: np.array, args) -> np.array:
    """量子化行列を用いてクオリティ q で圧縮"""

    T = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float64)

    Q = T.copy()
    if args.quality < 50:
        Q *= 50 / args.quality
    else:
        Q *= (100 - args.quality) / 50
    Q = np.round(Q).astype(np.uint32)

    d = dct.copy()
    for i in range(dct.shape[0]):
        for j in range(dct.shape[1]):
            d[i][j] = round(d[i][j] / Q[i][j]) * Q[i][j]
    return d


def peak_to_peak_signal_to_noise_ratio(image1: Image,
                                       image2: Image) -> float:
    """二画像の PSNR を計算する"""

    assert image1.size == image2.size
    assert image1.format == image2.format
    size = image1.size

    eps = 0
    for (x, y) in product(range(image1.size[0]), range(image2.size[1])):
        pixel1, pixel2 = image1.getpixel((x, y)), image2.getpixel((x, y))

        eps += (pixel1 - pixel2) ** 2

    if eps == 0:
        return 1

    eps /= size[0] * size[1]
    psnr = 10 * log10(255 * 255 / eps)

    return psnr


def main() -> NoReturn:
    parser = ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('-m', '--method', type=int, default=1)
    parser.add_argument('-t', '--threshold', type=int, default=None)
    parser.add_argument('-s', '--small', type=int, default=None)
    parser.add_argument('-q', '--quality', type=int, default=None)
    args = parser.parse_args()

    image = Image.open(args.source).convert(mode='L')
    image = expand_image(image)

    method, parameter = method_1, None
    if args.method == 2:
        assert args.threshold is not None
        method = method_2
        parameter = args.threshold

    if args.method == 3:
        assert args.small is not None
        assert 0 <= args.small <= 100
        method = method_3
        parameter = args.small

    if args.method == 4:
        assert args.small is not None
        assert 0 <= args.small <= 100
        method = method_4
        parameter = args.small

    if args.method == 5:
        assert args.quality is not None
        assert 0 < args.quality < 100
        method = method_5
        parameter = args.quality

    d1 = dct(image, method, args)
    d2 = idct(d1)

    d2.save(f'result_{args.method}_{parameter}.png')

    print(f'image: {args.source}')
    print(f'method: {args.method}(parameter = {parameter})')
    print(f'PSNR = {peak_to_peak_signal_to_noise_ratio(image, d2)}')


if __name__ == '__main__':
    main()
