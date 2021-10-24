from PIL import Image
from typer import Typer, Option, Exit, secho, colors
from typing import Optional, Tuple
from pathlib import Path
from itertools import product
from math import log10

app = Typer(add_completion=False)


@app.command("simple_read_bw")
def simple_load_and_show_and_save_grayscale_image(
    src_file: str = Option(..., '-S', '--source',
                           help='Source file of a grayscale image'),
    dst_file: str = Option(..., '-D', '--destination',
                           help='Destination file of the source image'),
) -> None:
    """src_file を load して show して dst_file に save する。"""

    if not (home / src_file).exists():
        secho("! The source image doesn't exist", fg=colors.RED)
        raise Exit()

    with Image.open(src_file) as image:
        image.show()
        image.save(dst_file)
        secho(f"+ The image saved at {dst_file}", fg=colors.GREEN)


@app.command('simple_color_change')
def simple_color_change(
    src_file: str = Option(..., '-S', '--source',
                           help='Source file of a color image'),
    dst_file: str = Option(..., '-D', '--destination',
                           help='Destination file of the generated image'),
    x_color: str = Option(..., '-X', '--xcolor',
                          help='color component X[r, g, b]'),
    y_color: str = Option(..., '-Y', '--ycolor',
                          help='color component Y[r, g, b]')
) -> None:
    """src_file の x_color 色成分と y_color 色成分を入れ替えた RGB 画像を dst_file として保存する"""

    if not (home / src_file).exists():
        secho("! The source image doesn't exist", fg=colors.RED)
        raise Exit()

    def normalize_color(color: str) -> Optional[int]:
        color = color.lower()[0]
        try:
            return ['r', 'g', 'b'].index(color)
        except:
            return None

    x_color, y_color = normalize_color(x_color), normalize_color(y_color)
    if x_color is None or y_color is None:
        secho("! The color expression is illegal", fg=colors.RED)
        raise Exit()

    matrix = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
    matrix[x_color], matrix[y_color] = matrix[y_color], matrix[x_color]
    matrix = sum(matrix, ())

    with Image.open(src_file) as image:
        image.convert("RGB", matrix).save(dst_file)
        secho(f"+ The generated image saved at {dst_file}", fg=colors.GREEN)


@app.command('simple_mix')
def simple_mix_two_grayscale_images(
    src_files: Tuple[str, str] = Option(..., '-S', '--source',
                                        help='source files of a grayscale image'),
    dst_file: str = Option(..., '-D', '--destination',
                           help='Destination file of the generated image'),
    alpha: float = Option(..., '-A', '--alpha', help='Alpha rate[0.0, 1.0]'),
) -> None:
    for src_file in src_files:
        if not (home / src_file).exists():
            secho("! The source image doesn't exist", fg=colors.RED)
            raise Exit()

    with Image.open(src_files[0]) as image1, \
            Image.open(src_files[1]) as image2:
        Image.blend(image1, image2, alpha).save(dst_file)
        secho(f"+ The generated image saved at {dst_file}", fg=colors.GREEN)


@app.command("PSNR")
def peak_to_peak_signal_to_noise_ratio(
    src_files: Tuple[str, str] = Option(...,
                                        '-S', '--source', help='source files'),
) -> None:
    """二画像の PSNR を計算する"""

    for src_file in src_files:
        if not (home / src_file).exists():
            secho("! The source image doesn't exist", fg=colors.RED)
            raise Exit()

    with Image.open(src_files[0]) as image1, \
            Image.open(src_files[1]) as image2:
        if image1.size != image2.size:
            secho("! The source images are not same size", fg=color.RED)
            raise Exit()
        size = image1.size

        if image1.format != image2.format:
            secho("! The source images are not same format", fg=color.RED)
            raise Exit()

        eps = 0
        dimention = 1 if isinstance(image1.getpixel(
            (0, 0)), int) else len(image1.getpixel((0, 0)))
        for (x, y) in product(range(size[0]), range(size[1])):
            pixel1, pixel2 = [image1.getpixel((x, y))], [
                image2.getpixel((x, y))]
            if isinstance(pixel1, int):
                pixel1, pixel2 = [pixel1], [pixel2]

            for i in range(len(pixel1)):
                eps += pow(pixel1[i] - pixel2[i], 2)

        if eps == 0:
            secho("+ The source images are same", fg=colors.GREEN)
            raise Exit()
        eps /= size[0] * size[1] * dimention
        psnr = 10 * log10(255 * 255 / eps)

        secho(f"+ PSNR = {psnr}", fg=colors.GREEN)


if __name__ == "__main__":
    app()
