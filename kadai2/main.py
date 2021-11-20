import seaborn as sns
from argparse import ArgumentParser
from PIL import Image
from typing import NoReturn, List


def image_to_list(image: Image) -> List[int]:
    size, data = image.size, []
    for x in range(size[0]):
        for y in range(size[1]):
            data.append(image.getpixel((x, y)))
    return data


def calc_variance(data: List[int]) -> NoReturn:
    mean = sum(data) / len(data)

    variance = 0
    for e in data:
        variance += (e - mean) ** 2
    return variance / len(data)


def make_histogram(data: List[int], out: str) -> NoReturn:
    plot = sns.histplot(data, discrete=True)
    plot.get_figure().savefig(out)


def make_diffimage(image: Image, predictor) -> List[int]:
    size, offset = None, None
    try:
        predictor(1, 0)
    except:
        size = (image.size[0] - 1, image.size[1] - 1)
        offset = (1, 1)
    else:
        size = (image.size[0] - 1, image.size[1])
        offset = (0, 1)

    diffimage = []
    for _x in range(size[0]):
        for _y in range(size[1]):
            x, y = _x + offset[0], _y + offset[1]

            pixel = image.getpixel((x, y))
            predict = predictor(x, y)
            diff = pixel - predict

            diffimage.append(diff)
    return diffimage


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('-p', '--predictor', default='equal')
    args = parser.parse_args()

    with Image.open(args.image) as image:
        # 原画像
        print(f'variance_origin = {calc_variance(image_to_list(image))}')
        make_histogram(image_to_list(image), 'histogram_origin.png')

        def predictor_equal(x: int, y: int) -> int:
            assert y > 0

            return image.getpixel((x, y - 1))

        def predictor_even(x: int, y: int) -> int:
            assert x > 0 and y > 0

            return image.getpixel((x - 1, y)) + image.getpixel((x, y - 1)) - image.getpixel((x - 1, y - 1))

        def predictor_odd(x: int, y: int) -> int:
            assert x > 0 and y > 0

            return round((image.getpixel((x - 1, y)) + image.getpixel((x, y - 1))) / 2)

        predictor = predictor_equal
        if args.predictor == 'even':
            predictor = predictor_even
        if args.predictor == 'odd':
            predictor = predictor_odd

        # 差分画像
        diffimage = make_diffimage(image, predictor)
        print(f'variance_diff = {calc_variance(diffimage)}')
        make_histogram(diffimage, 'histogram_diff.png')
