from argparse import ArgumentParser
from typing import NoReturn
from PIL import Image
import numpy as np
import seaborn as sns


def make_histogram(data: np.array, out: str) -> NoReturn:
    plot = sns.histplot(data, discrete=True)
    plot.get_figure().savefig(out)


def main() -> NoReturn:
    parser = ArgumentParser()
    parser.add_argument('first')
    parser.add_argument('second')
    args = parser.parse_args()

    frame_1 = np.array(Image.open(args.first).convert(mode='L'), dtype=int)
    frame_2 = np.array(Image.open(args.second).convert(mode='L'), dtype=int)
    diff = frame_1 - frame_2

    # 差分画像の作成・保存
    Image.fromarray(np.absolute(diff).astype(np.uint8)).save('diff.png')

    # 差分画像のヒストグラムの作成・保存
    make_histogram(diff.flatten(), 'diff_histogram.png')


if __name__ == '__main__':
    main()
