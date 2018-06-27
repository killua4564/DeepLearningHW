import os
import random
import datetime
from gensim.models import word2vec

E = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job': 3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

def bulid_word2vec():
	sentences = [open('./testing/{index}.txt'.format(index=str(index)), 'r', encoding='utf-8').read().strip().split(' / ') for index in range(1000)]
	for i in E.keys():
		for j in os.listdir('./training/{key}_cut/'.format(key=i)):
			sentences.append(open("./training/{key}_cut/{file}".format(key=i, file=j), 'r', encoding='utf-8').read().split(' / '))
	random.shuffle(sentences)
	for min_count in range(1, 10):
		word2Vec = word2vec.Word2Vec(sentences, min_count=min_count, size=270, iter=10, sg=1, workers=10)
		word2Vec.save("word2vec{count}.model".format(count=min_count))

def print_word2vec():
	for index in range(1, 10):
		model = word2vec.Word2Vec.load('word2vec{index}.model'.format(index=index))
		print(index, len(model.wv.index2word))

if __name__ == '__main__':
	# bulid_word2vec()
	# print_word2vec()

	model = word2vec.Word2Vec.load('word2vec2.model')
	print(model.vector_size)
	# print(len(model.wv.index2word))
	print(model.most_similar('日本'))
	print(model.similarity('日本', '台灣'))