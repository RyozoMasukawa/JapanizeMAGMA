# JapanizeMAGMA
MAGMAを日本語化したもの

rinna社の[GPT2](https://huggingface.co/rinna/japanese-gpt2-medium)を利用して画像言語モデル[MAGMA](https://github.com/Aleph-Alpha/magma)を日本語化しました!

## 訓練手順[MAGMA](https://github.com/Aleph-Alpha/magma)のREADMEの翻訳

### データセットの定義
```.python
from magma.datasets.convert_datasets import convert_dataset

def my_dataset_iterator():
    """
    画像のパスとキャプション, メタデータをセットにしたタプルをyieldするイテレータを実装してください
    image_path, {"captions": [...], "metadata": {...}, }, image_pathはPathオブジェクトのインスタンスとなっている画像のパス名, captionは各画像のキャプションの文字列のリスト, metadataはオプションです
    """

if __name__ == "__main__":
    convert_dataset(data_dir="/target/directory", ds_iterator=my_dataset_iterator())
```

### コンフィグを指定して訓練を開始してください
```.bash
deepspeed train.py --config path_to_my_config
```


アナハイムはやりやがったってことだ
