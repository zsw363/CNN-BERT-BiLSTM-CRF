# -*- coding utf-8 -*-
"""
Create on 2021/3/15 17:31
@author: zsw
"""
import random
import pickle
import sklearn

import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import  ClassifierI
from statistics import mode

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

#

exit()


#自定义数据集的情感分析
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("short_reviews/positive.txt", 'r').read()
short_neg = open("short_reviews/negative.txt", 'r').read()
documents = []
for r in short_pos.split('\n'):
    documents.append((r, 'pos'))
for r in short_neg.split('\n'):
    documents.append((r, 'neg'))
all_words = []
short_pos_words = nltk.word_tokenize(short_pos)
short_neg_words = nltk.word_tokenize(short_neg)
for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words['stupid'])

#词语出现
word_features = list(all_words.keys())[:5000]
def find_features(document):
    words = nltk.word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)
#训练
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Naive Vayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#sklearn
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

exit()


#统计&情感分析
#不同分类器投票
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words['stupid'])

#词语出现
word_features = list(all_words.keys())[:3000]
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

#训练
training_set = featuresets[:1900]
testing_set = featuresets[1900:]
#训练
# training_set = featuresets[100:]
# testing_set = featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(training_set)


print("Naive Vayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#sklearn
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
#GaussianNB, BernoulliNB
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accuracy:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

#不同分类器投票

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,)
print("voted_classifier accuracy:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classification:", voted_classifier.classify(testing_set[0][0]),
      "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]),
      "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]),
      "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
exit()

#保存为pickle文件
save_classifier = open("naivebayes.pickle",'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()

classifier_f = open('naivebayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
exit()



#词网
syns = wordnet.synsets('program')
print(syns)
print(syns[0])
print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

synonyms = []
antonyms = []
for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        # print('l:', l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))
#相似度
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cactus.n.01')
print(w1.wup_similarity(w2))

exit()


#corpus语料库导入
sample = gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(sample)
print(tok[:10])
exit()


#词形还原
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos='a'))
print(lemmatizer.lemmatize("best", pos='a'))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("ran", 'v'))
exit()


#词性标签

# CC     coordinatingconjunction 并列连词
# CD     cardinaldigit  纯数  基数
# DT     determiner  限定词（置于名词前起限定作用，如 the、some、my 等）
# EX     existentialthere (like:"there is"... think of it like "thereexists")存在句；存现句
# FW     foreignword  外来语；外来词；外文原词
# IN     preposition/subordinating conjunction介词/从属连词；主从连词；从属连接词
# JJ     adjective    'big'  形容词
# JJR    adjective, comparative 'bigger' （形容词或副词的）比较级形式
# JJS    adjective, superlative 'biggest'  （形容词或副词的）最高级
# LS     listmarker  1)
# MD     modal (could, will) 形态的，形式的 , 语气的；情态的
# NN     noun, singular 'desk' 名词单数形式
# NNS    nounplural  'desks'  名词复数形式
# NNP    propernoun, singular     'Harrison' 专有名词
# NNPS  proper noun, plural 'Americans'  专有名词复数形式
# PDT    predeterminer      'all the kids'  前位限定词
# POS    possessiveending  parent's   属有词  结束语
# PRP    personalpronoun   I, he, she  人称代词
# PRP$  possessive pronoun my, his, hers  物主代词
# RB     adverb very, silently, 副词    非常  静静地
# RBR    adverb,comparative better   （形容词或副词的）比较级形式
# RBS    adverb,superlative best    （形容词或副词的）最高级
# RP     particle     give up 小品词(与动词构成短语动词的副词或介词)
# TO     to    go 'to' the store.
# UH     interjection errrrrrrrm  感叹词；感叹语
# VB     verb, baseform    take   动词
# VBD    verb, pasttense   took   动词   过去时；过去式
# VBG    verb,gerund/present participle taking 动词  动名词/现在分词
# VBN    verb, pastparticiple     taken 动词  过去分词
# VBP    verb,sing. present, non-3d     take 动词  现在
# VBZ    verb, 3rdperson sing. present  takes   动词  第三人称
# WDT    wh-determiner      which 限定词（置于名词前起限定作用，如 the、some、my 等）
# WP     wh-pronoun   who, what 代词（代替名词或名词词组的单词）
# WP$    possessivewh-pronoun     whose  所有格；属有词
# WRB    wh-abverb    where, when 副词

train_text = state_union.raw('2005-GWBush.txt')
sample_text = "PRESIDENT GEORGE W. BUSH'S ADDRESS BEFORE " \
              "A JOINT SESSION OF THE CONGRESS ON THE STATE " \
              "OF THE UNION"
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
def process_content():
    try:

        """
        NE              Type Examples
        ORGANIZATION    Georgia-Pacific Corp., WHO
        PERSON          Eddy Bonte, President Obama
        LOCATION        Murray River, Mount Everest
        DATE            June, 2008-06-29
        TIME            two fifty a.m., 1:30 p.m. 
        MONEY           175 million Canadian Dollars, GBP 10.40
        PERCENT         twenty pct, 18.75%
        FACILITY        Washington Monument, Stonehenge
        GPE             South East Asia, Midlothian
        """

        for i in tokenized[:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            #只找命名实体
            nameEnt = nltk.ne_chunk(tagged, binary=True)
            #命名实体类别细分
            nameEnt = nltk.ne_chunk(tagged)

            nameEnt.draw()
        exit()


        for i in tokenized[:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<.*>+}
                            }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
        exit()


        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # print(tagged)
            exit()


            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            # print(chunked)
            exit()

    except Exception as e:
        print(e.__class__.__name__)
        print(str(e))

process_content()



# PorterStemmer分词算法 pythoning-python
ps = PorterStemmer()

example_words = ['python','pythoning','pythoner','pythoned','pythonly']
for w in example_words:
    print(ps.stem(w))
exit()


# 停用词
stoplist = stopwords.words('english')
print(stoplist)
exit()

