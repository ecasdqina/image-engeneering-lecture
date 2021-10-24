# 使い方

## simple_read_bw

画像を読み込み、表示して、保存する。

```shell
python3 main.py simple_read_bw -S src.png -D dst.png
```

## simple_color_change

画像の x, y 色成分を交換した画像を作成する。

```shell
python3 main.py simple_color_change -S src.png -D dst.png -X r -Y g
```

## simple_mix

二画像を重ねた画像を作成する。

```shell
python3 main.py simple_mix -S A.png B.png -D dst.png -A 0.5 
```

## PSNR

二画像の PSNR を計算する。

```shell
python3 main.py PSNR -S A.png B.png
```
