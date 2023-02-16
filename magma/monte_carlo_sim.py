import pandas as pd
import numpy as np
import re
import torch

import MeCab
from tqdm import tqdm 
from magma import Magma
from magma.image_input import ImageInput

class MagmaMonteCarloSimulator:
    def __init__(self, model : Magma, img_urls : list, titles:list=None, batch_size=16):
        if titles is not None:
            assert len(img_urls) == len(titles), "length of urls and titles must be the same!"
        
        self.model = model
        self.img_urls = img_urls
        self.titles = titles
        self.tagger = MeCab.Tagger("-Owakati")
        self.batch_size = batch_size
        
    def get_result(self, prompt : str, top_n:int=50, sim_step:int=100, need_captions:bool=False):
        df_item_caption = self.__simulate_magma(prompt, sim_step=sim_step)
        whole_word_list, word_sets, bows, big_bow = self.__get_document_info(df_item_caption) 
        tfidf = self.__tfidf_calculator(whole_word_list, word_sets, bows)
        tf_idf_each_doc = self.__get_each_feature(tfidf, bows, whole_word_list)
        if self.titles is not None:
            top_n_tf_idf = [[self.img_urls[idx], self.titles[idx], sorted(tf_idf_each_doc[idx], key=lambda t: t[1], reverse=True)[:top_n]] for idx in range(len(tf_idf_each_doc))]
        else:
            top_n_tf_idf = [[self.img_urls[idx], sorted(tf_idf_each_doc[idx], key=lambda t: t[1], reverse=True)[:top_n]] for idx in range(len(tf_idf_each_doc))]
            
        if need_captions:
            return top_n_tf_idf, self.__combine_all_text(df_item_caption)
        return top_n_tf_idf
    
    #urlに指定された全ての画像のMAGMAのキャプションを得る関数
    def __get_caption_list(self, prompt, img_url_list, max_steps=40):
        caption_list = []
        for url in tqdm(img_url_list):
            inputs =[
                ## supports urls and path/to/image
                ImageInput(url),
                prompt
            ]
            embeddings = self.model.preprocess_inputs(inputs)  
            output = self.model.generate(
                embeddings = embeddings,
                max_steps = max_steps,
                temperature = 0.7,
                # top_k = 0,
                # top_p = 0.9
            )
            caption_list.append(output[0])
        assert len(img_url_list) == len(caption_list)
        return caption_list
    
    #urlに指定された全ての画像のMAGMAのキャプションを得る関数(バッチ処理)
    def __get_caption_list_batch(self, prompt, img_url_list, max_steps=40, show_pbar=True, batch_size=32):
        caption_list = []
        pbar = tqdm(img_url_list) if show_pbar else img_url_list
        input_batch = []
        count = 0
        list_of_embeddings = []

        for url in pbar:
            inputs = [
                ## supports urls and path/to/image
                ImageInput(url),
                prompt
            ]
        
        
            with torch.no_grad():
                embeddings = self.model.preprocess_inputs(inputs)  
                list_of_embeddings.append(embeddings)
                count += 1

                if count % batch_size == 0 or count == len(img_url_list):
                    embeddings_batch = torch.cat(list_of_embeddings)
                    outputs = self.model.generate(
                        embeddings = embeddings_batch,
                        max_steps = max_steps,
                        temperature = 0.7,
                        # top_k = 0,
                        # top_p = 0.9
                    )
                    list_of_embeddings = []
                    caption_list += outputs
        assert len(img_url_list) == len(caption_list)
        return caption_list
        
    #MAGMAにsim_steps回出力させて、そのキャプションの文書を得るメソッド
    def __simulate_magma(self, prompt, sim_step=100):
        urls = self.img_urls
        list_of_captions = []
        for _ in range(sim_step):
            captions = self.__get_caption_list_batch(prompt, self.img_urls, batch_size=self.batch_size)
            list_of_captions.append(captions)
        caption_array = np.array(list_of_captions)

        item_array_list = []
        for lst in caption_array.T:
            try:
                sample_list_re = [re.match(r"([^。]+)", txt).group(0) for txt in lst]
                item_array_list.append(sample_list_re)
            except:
                continue
        item_array = np.array(item_array_list)

        dict_for_df_item = {}
        for i in range(item_array.shape[0]):
            dict_for_df_item[urls[i]] = item_array[i]
        df_item_caption = pd.DataFrame(dict_for_df_item)

        return df_item_caption

    def __split_word(self, txt : str) -> list:
        return self.tagger.parse(txt).split(" ")

    def __combine_all_text(self, df_item_caption : pd.DataFrame):
        df_item_caption_T = df_item_caption.T

        text_array = np.array(df_item_caption_T)
        texts = ""

        for i in range(text_array.shape[0]):
            texts += " ".join(text_array[i])
        return texts
    
    #tfidf算出に必要な基本的な情報を文書から得る
    def __get_document_info(self, df_item_caption_30: pd.DataFrame):
        bows = []
        word_sets = []
        whole_word_set = set()
        big_bow = {}
        for i in tqdm(range(df_item_caption_30.shape[1])):
            img_url = df_item_caption_30.columns[i]
            sample_texts = df_item_caption_30.iloc[:, i].apply(self.__split_word)
            bow = {}
            word_set = set()
            for sample_text in sample_texts:
                for word in sample_text:
                    whole_word_set.add(word)
                    word_set.add(word)
                    if word not in bow.keys(): 
                        bow[word] = 1 
                        big_bow[word] = 1
                    else: 
                        bow[word] += 1
                        big_bow[word] += 1
            bows.append(bow)
            word_sets.append(word_set)
        return list(whole_word_set), list(map(list, word_sets)), bows, big_bow
    
    #全ての文書のtfidfを出す関数
    def __tfidf_calculator(self, whole_word_set, word_sets, bows):
        N = len(bows)
        tf = {}
        df = dict.fromkeys([i for i in range(len(whole_word_set))], 0)
        for i, word in enumerate(whole_word_set):
            for j in range(N):
                if word in word_sets[j]:
                    df[i] += 1
                    tf[(i, j)] = bows[j][word]

        #得られたtf, dfからtfidfを計算
        tfidf = {}
        for i, word in enumerate(whole_word_set):
            for j in range(N):
                if word in word_sets[j]:
                    tfidf[(i, j)] = tf[(i, j)] * np.log(N / df[i])
                else:
                    tfidf[(i, j)] = 0
        return tfidf
    
    #各デスクリプションの各単語とそのtfidfのペアのリストを得る関数
    def __get_each_feature(self, tfidf, bows, whole_word_list):
        tf_idf_each_doc = []
        for j, bow in enumerate(bows):
            tf_idf_list = []
            for word in bow.keys():
                i = whole_word_list.index(word)
                tf_idf_list.append((word, tfidf[i, j]))
            tf_idf_each_doc.append(tf_idf_list)
        return tf_idf_each_doc
