## インストール
* 一応ubuntu20.04, noetic, python3.8で動作確認しています. 他の環境で動くかはわかりませんが問題あったら教えてください.

mohou_rosのインストール手順にしたがって, mohou_rosをインストールしてください. mohou rosのページにも書いていますが普通のrosのインストールと違って`pip install -e .`もしないといけないので注意してください.
実行はeusからやる(自分でラッパーを書く)と思うので, ros packageとしてインストールする必要はないです. その場合は, 以下のrosrun mohou hoge.py部分をpython3 hoge.pyに置き換えてやってみてください.

## データ準備
```
mkdir -p ~/.mohou/iwata  # このdirectryに学習データ等が保存される
mkdir -p ~/.mohou/iwata/rosbag  # このdirectryに.bagファイルをおいて
```
ここで ~/.mohou/iwataをmohouおよびmohou rosでは"project path"と読んでいて, "iwata"をproject nameと読んでいます. この名前は任意です. 本ソフトウェア内の ./dataset_generation.pyや ./tune_image_filter.py内ではハードコードされていますが, 変更してもらって構いませんん. ソフトウェア内では project pathは pp , project nameは pn と略されることが多いです.


イメージフィルタ (crop, blur, resize)のチューニング. resolutionサイズは選択できるようにはなっているが, 必ず112にしてください (112 x 112)の画像に変換されます). そうしないと autoencoderの学習時にエラーが出ます. チューニング結果は "~/.mohou/iwata/image_config.yaml"に保存されます. これは学習データセット生成にも使われますし, 実行時にも使われます. (学習時と実行時でimage filterの設定が異なると当然うまくいきません)
```
python3 tune_image_filter.py
```
*qを押すと終了します.*

次にこのデータを学習データセットに変換します.
```
python3 ./dataset_generation.py
```

### 訓練
`mohou_ros`がros packageとして入っていない場合でも, 直接python スクリプトとして実行できます.
```bash
rosrun mohou_ros train.py -pn iwata -n_vae 1500 -n_lstm 20000
# python3 ~/mohou_ws/src/mohou_ros/scripts/train.py -pn iwata -n_vae 1500 -n_lstm 20000  # ros pkacgeとしてmohou rosをいれてないならこれでもいい
```
ここでのpn はproject nameの略です.

### 実行
だいたい ./executor.py のような感じです. dummyな関数部分を適当にsubscriberに変えたりしてください. 
