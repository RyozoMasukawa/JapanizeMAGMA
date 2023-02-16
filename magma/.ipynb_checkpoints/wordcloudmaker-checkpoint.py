import codecs
from wordcloud import WordCloud
import re
import MeCab


class WordCloudMaker: 
    def __init__(self,text=None,font_path=None,width=800,height=600,min_font_size=15):
        """
        コンストラクタ
        """
        self.font_path = font_path         # フォントのパス
        self.text = text                   # クラウド化したいテキスト
        self.background_color = 'white'    # 画像の背景色
        self.width = width                 # 画像の横ピクセル数
        self.height = height               # 画像の縦ピクセル数
        self.min_font_size = min_font_size # 最小のフォントサイズ
        self.tagger = MeCab.Tagger("-Owakati")
    
    def split_word(self, txt : str) -> list:
        return self.tagger.parse(txt).split(" ")

    def create(self,path,exclusion=[]):
        """
          ワードクラウドの画像生成

        Parameters:
            path : str         画像の出力パス
            exclusion : [str]  除外ワードのリスト
        """      
        # 名詞の抽出
        words = self.extract_words(self.text,exclusion)
        # ワードクラウドの画像生成
        words = self.generate_wordcloud(path,words)

    def generate_wordcloud(self,path,words):
        """
        ワードクラウドの画像生成

        Parameters:
            path : str        画像の出力パス
            words : [str]     ワードクラウド化したい名詞リスト
        """
        #ワードクラウドの画像生成
        wordcloud = WordCloud(
                background_color=self.background_color, # 背景色 
                font_path=self.font_path,               # フォントのパス
                width=self.width,                       # 画像の横ピクセル数
                height=self.width,                      # 画像の縦ピクセル数
                min_font_size=self.min_font_size,       # 最小のフォントサイズ
                max_words=400,
            )
        # ワードクラウドの作成
        wordcloud.generate(words)
        # 画像保存
        wordcloud.to_file(path) 

    def extract_words(self,text,exclusion=[]):
        """
        形態素解析により一般名詞と固有名詞のリストを作成
        ---------------
        Parameters:
            text : str         テキスト
            exclusion : [str]  除外したいワードのリスト
        """
        words = self.split_word(text)
        
        words_final = []
        for word in words:
            if word not in exclusion:
                words_final.append(word)
    
        return ' ' . join(words_final)

    def read_file(self,filename):
        '''
        ファイルの読み込み

        Parameters:
        --------
            filename : str   要約したい文書が書かれたファイル名 
        '''
        with codecs.open(filename,'r','utf-8','ignore') as f:
            self.read_text(f.read())

    def read_text(self,text):
        '''
        テキストの読み込み
        '''
        self.text = text
