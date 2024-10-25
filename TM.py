import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import hdbscan
import umap.umap_ as umap
from transformers import BertTokenizer, BertModel
from langdetect import detect
from Preprocess import getText
import numpy as np
import random
import torch


class Topic_Model(object):
    def __init__(self, k=10, method='LDA', num_keywords=20, seed=100):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'BERT_LDA', 'BERT_LDA_Kmeans', 'BERT_TFIDF_HDBSCAN', 'BERT_LDA_HDBSCAN', 'BERT_TFIDF_Kmeans'}:
            raise Exception('Invalid method!')
        print('Initialized')
        self.k = k
        self.stopwords = None
        self.seed = seed
        self.cluster_model = None
        self.ldamodel = None
        self.gamma = 15  # parameter for reletive importance of lda
        self.vec = {}
        self.method = method
        self.AE = None
        self.num_keywords = num_keywords
        
    def set_random_seed(self):
        np.random.seed(self.seed)  # Use self.seed directly here
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


       
    def vectorize(self, method=None):
        from gensim import corpora
        from sklearn.feature_extraction.text import TfidfVectorizer
        import gensim
        from transformers import AutoTokenizer, AutoModel
        import pickle
        import torch
        import numpy as np
        from torch.utils.data import Dataset, DataLoader

        """Get vector representations from selected methods"""
        dictionary = corpora.Dictionary.load('dictionary')
        corpus = corpora.MmCorpus('corpus')
        with open("x_train_rus", "rb") as fp:   # Unpickling
            x_train_rus = pickle.load(fp)
        if method is None:
            method = self.method

        elif method == 'TFIDF':
          print('Getting vector representations for TF-IDF ...')
          tfidf = TfidfVectorizer()
          vec = tfidf.fit_transform(x_train_rus)
          print('Getting vector representations for TF-IDF. Done!')
          return vec

        elif method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = gensim.models.LdaMulticore(corpus, num_topics=self.k, 
                                       id2word = dictionary,
                                       workers = 2, passes=10,
                                       chunksize=100, random_state=self.seed)
                def get_vec_lda(model, corpus, k):
                  n_doc = len(corpus)
                  vec_lda = np.zeros((n_doc, k))
                  for i in range(n_doc):
                      # get the distribution for the i-th document in corpus
                      for topic, prob in model.get_document_topics(corpus[i]):
                          vec_lda[i, topic] = prob
                  return vec_lda
                vec = get_vec_lda(self.ldamodel, corpus, self.k)
                print('Getting vector representations for LDA. Done!')
                return vec

        elif method == 'BERT':
            print('Getting vector representations for BERT ...')
            #for russian language
            lang = detect(x_train_rus[0])
            if lang == 'ru':
                tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
                model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
                model = BertModel.from_pretrained("bert-base-multilingual-cased")                          
            model= model.cpu() 
            x_train_rus_clear = []
            for i in x_train_rus:
              x_train_rus_clear.append(str(i))
            def embed_bert_cls(text, model, tokenizer):
              t = tokenizer(text, padding=True, truncation=False, return_tensors='pt')
              with torch.no_grad():
                  model_output = model(**{k: v.to(model.device) for k, v in t.items()})
              embeddings = model_output.last_hidden_state[:, 0, :]
              embeddings = torch.nn.functional.normalize(embeddings)
              return embeddings.cpu().numpy()
            class InterviewDataset(Dataset):
              def __init__(self, queries):
                  self.queries = queries
              def __len__(self):
                  return len(self.queries)
              def __getitem__(self, idx):
                  return str(self.queries[idx])
            data_loader = DataLoader(InterviewDataset(x_train_rus_clear), batch_size=1, shuffle=False)
            vecs = []
            for batch in enumerate(data_loader):
                batch_emdg = embed_bert_cls(batch[1], model, tokenizer)
                vecs.append(batch_emdg)
            vecs_bert = np.concatenate(vecs, axis=0 )
            print('Getting vector representations for BERT. Done!')
            return vecs_bert

    def fit(self, corpus, dictionary, method=None, cluster_model=None):
        from gensim import corpora
        import gensim
        import itertools
        from gensim.models import CoherenceModel
        import pickle
        import numpy as np
        self.set_random_seed()

        dictionary = corpora.Dictionary.load('dictionary')
        corpus = corpora.MmCorpus('corpus')  
        with open("x_train_rus", "rb") as fp:   # Unpickling
            x_train_rus = pickle.load(fp)     
        x_train_rus_clear = []
        for i in x_train_rus:
          x_train_rus_clear.append(str(i))        
        
        # Default method
        if method is None:
            method = self.method
        if method == 'LDA':
          if not self.ldamodel:
              print('Fitting LDA ...')
              self.ldamodel = gensim.models.LdaMulticore(corpus, num_topics=self.k, 
                                       id2word = dictionary,
                                       workers = 2, passes=10,
                                       random_state=self.seed,
                                       chunksize=100)
              
              print('Fitting LDA Done!')
              for idx, topic in self.ldamodel.print_topics(-1):
                print('Topic: {} Word: {}'.format(idx, topic))

              for i in range(len(x_train_rus_clear)):
                  x_train_rus_clear[i] = str(x_train_rus_clear[i]).split(' ')
              processed_docs = np.array(x_train_rus_clear)
              
              def compute_coherence(lda_model, dictionary, coherence_metrics):
                coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs,
                                                dictionary=dictionary, coherence=coherence_metrics)
                return coherence_model_lda.get_coherence()
              num_topics = self.k
              topic_words = []
              for i in range(num_topics):
                  tt = self.ldamodel.get_topic_terms(i, self.num_keywords)
                  topic_words.append([dictionary[pair[0]] for pair in tt])
              def topic_diversity(topic_words):
                  topk = 10
                  if topic_words is None:
                      return 0
                  if topk > len(topic_words[0]):
                      raise Exception('Words in topics are less than ' + str(topk))
                  else:
                      unique_words = set()
                      for topic in topic_words:
                          unique_words = unique_words.union(set(topic[:topk]))
                      td = len(unique_words) / (topk * len(topic_words))
                      return td
              def _LOR(P, Q):
                  lor = 0
                  for v, w in zip(P, Q):
                      if v > 0 or w > 0:
                          lor = lor + np.abs(np.log(v) - np.log(w))
                  return lor / len(P)
              def Kullback_Leibler(ldamodel):
                  beta = ldamodel.get_topics()
                  kl_div = 0
                  count = 0
                  for i, j in itertools.combinations(range(len(beta)), 2):
                      kl_div += _LOR(beta[i], beta[j])
                      count += 1
                  return kl_div / count

            #   print('_________________________________________________________________')
            #   print('C_v coherence: ', compute_coherence(self.ldamodel, dictionary, 'c_v'))
            #   print('U_mass coherence: ', compute_coherence(self.ldamodel, dictionary, 'u_mass'))
            #   print('UCI : ', compute_coherence(self.ldamodel, dictionary, 'c_uci'))
            #   print('NPMI : ', compute_coherence(self.ldamodel, dictionary, 'c_npmi'))
            #   print('Topic_diversity :', topic_diversity(topic_words))
            #   print('Kullback-Leibler Divergence :', Kullback_Leibler(self.ldamodel))
              return topic_words, self.ldamodel.print_topics(-1)
                
        # Default clustering method
        elif cluster_model is None:
            cluster_model = self.cluster_model

        elif cluster_model == 'Kmeans':

          print('Clustering embeddings ...')
          cm = KMeans(self.k, random_state=self.seed)
          self.vec[method] = self.vectorize(method)
          cm.fit(self.vec[method])
          print('Clustering embeddings. Done!')
          print('Getting topic words')
          def get_topic_words(token_lists, labels, k=None):
            """get top words within each topic from clustering results"""
            if k is None:
                k = len(np.unique(labels))
            topics = ['' for _ in range(k)]
            for i, c in enumerate(token_lists):
                topics[labels[i]] += (' ' + ' '.join(c))
            word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
            # get sorted word counts
            word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
            # get topics
            topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))
            return topics

          def get_coherence(labels, token_lists, measure):
            topics = get_topic_words(token_lists, labels)
            cm = CoherenceModel(topics=topics, texts=token_lists, corpus=corpus, dictionary=dictionary,
                                    coherence=measure)
            return cm.get_coherence()
          with open("x_rus", "rb") as fp:   # Unpickling
              x_rus = pickle.load(fp)
          topics = get_topic_words(x_rus, cm.labels_)
          df_topic_keywords = pd.DataFrame(topics)
          df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
          df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    
          def topic_diversity(topic_words):
              topk = 10
              if topic_words is None:
                  return 0
              if topk > len(topic_words[0]):
                  raise Exception('Words in topics are less than ' + str(topk))
              else:
                  unique_words = set()
                  for topic in topic_words:
                      unique_words = unique_words.union(set(topic[:topk]))
                  td = len(unique_words) / (topk * len(topic_words))
                  return td

        #   print('_________________________________________________________________')
        #   print('C_v coherence: ', get_coherence(cm.labels_, x_rus, measure='c_v'))
        #   print('U_mass coherence: ', get_coherence(cm.labels_, x_rus, measure='u_mass'))
        #   print('UCI : ', get_coherence(cm.labels_, x_rus, measure='c_uci'))
        #   print('NPMI : ', get_coherence(cm.labels_, x_rus, measure='c_npmi'))
        #   print('Topic_diversity : ', topic_diversity(topics))
        #   print('_________________________________________________________________')

          return df_topic_keywords

        elif cluster_model == 'hdbscan':
          self.vec[method] = self.vectorize(method)
          np.random.seed(self.seed)
          umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, random_state=self.seed, 
                            metric='cosine').fit_transform(self.vec[method])
          cm = hdbscan.HDBSCAN(gen_min_span_tree=True, min_cluster_size=5, min_samples = 6,
                          metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
          docs_df = pd.DataFrame(x_train_rus_clear, columns=["Doc"])
          docs_df['Topic'] = cm.labels_
          docs_df['Doc_ID'] = range(len(docs_df))
          docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
          labels = list(docs_df['Topic'])

          #TRYING NEW METHOD, UNDER DEVELOPMENT
          print('UNDER DEVELOPMENT')
          docs_df.to_csv('clusters_documents.csv', index=False)
          print(docs_df)
          

          def c_tf_idf(documents, m, ngram_range=(1, 1)):
            with open("all_sw", "rb") as fp:   # Unpickling
                all_sw = pickle.load(fp)
            count = CountVectorizer(ngram_range=ngram_range, stop_words=all_sw).fit(documents)
            t = count.transform(documents).toarray()
            w = t.sum(axis=1)
            tf = np.divide(t.T, w)
            sum_t = t.sum(axis=0)
            idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
            tf_idf = np.multiply(tf, idf)

            return tf_idf, count
          
          tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(x_train_rus_clear))
          def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=self.num_keywords):
            words = count.get_feature_names()
            labels = list(docs_per_topic.Topic)
            tf_idf_transposed = tf_idf.T
            indices = tf_idf_transposed.argsort()[:, -n:]
            top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
            return top_n_words

          def extract_topic_sizes(df):
            topic_sizes = (df.groupby(['Topic'])
                            .Doc
                            .count()
                            .reset_index()
                            .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                            .sort_values("Size", ascending=False))
            return topic_sizes
          
          top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, self.num_keywords)
          topic_sizes = extract_topic_sizes(docs_df) 
          
          #topic reduction
          wanted = int(self.k)
          resize = len(topic_sizes) - (wanted+1)
          from sklearn.metrics.pairwise import cosine_similarity
          for i in range(resize):
            # Calculate cosine similarity
            similarities = cosine_similarity(tf_idf.T)
            np.fill_diagonal(similarities, 0)

            # Extract label to merge into and from where
            topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
            topic_to_merge = topic_sizes.iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Adjust topics
            docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            old_topics = docs_df.sort_values("Topic").Topic.unique()
            map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
            docs_df.Topic = docs_df.Topic.map(map_topics)
            docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

            # Calculate new topic words
            m = len(x_train_rus_clear)
            tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
            top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, self.num_keywords)
          
          topic_sizes = extract_topic_sizes(docs_df) 
          def get_topic_words(token_lists, labels, k=None):
              if k is None:
                  k = len(np.unique(labels))
              topics = ['' for _ in range(k)]
              for i, c in enumerate(token_lists):
                  topics[labels[i]] += (' ' + ' '.join(c))
              word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
              word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
              topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))
              return topics
        


          def extract_top_documents(docs_df, n):
                top_docs_per_topic = {}

                # Group by topic and rank documents within each topic
                for topic in docs_df['Topic'].unique():
                    if topic == -1:
                        continue  # Skip the noise (-1) cluster

                    # Filter documents belonging to the current topic
                    topic_docs = docs_df[docs_df['Topic'] == topic]

                    # Here, I'm using a simple length-based heuristic. 
                    # You might replace this with a more sophisticated ranking based on HDBSCAN probabilities or other metrics.
                    topic_docs['Doc_Length'] = topic_docs['Doc'].apply(len)
                    top_docs = topic_docs.nlargest(n, 'Doc_Length')

                    top_docs_per_topic[topic] = top_docs

                return top_docs_per_topic

            # Call the function to get the top documents for each topic
          df_raw = pd.read_csv('df_raw.csv')

          text = getText('text.txt')
          paragraphs = text.split('\n')
          df_raw = pd.DataFrame(paragraphs,
                            columns = ['paragraphs'],
                            index = range(1, len(paragraphs)+1))

        #   print(len(df_raw))
        #   print(len(x_train_rus))  
          with open("x_train_rus_alligned", "rb") as fp:   # Unpickling
            x_train_rus_alligned = pickle.load(fp)    
        #   print('x_train_rus_alligned', len(x_train_rus_alligned))
          top_documents_per_topic = extract_top_documents(docs_df, n=10)
        #   for topic, top_docs in top_documents_per_topic.items():
        #         print(f"Top documents for Topic {topic}:")
        #         for index, row in top_docs.iterrows(): #row['Doc'] - lemmatized phrase
        #             for i in range(len(x_train_rus_alligned)):
        #                 if x_train_rus_alligned[i] == row['Doc']:
        #                     print(f"Document: {df_raw['paragraphs'].iloc[i]}")  
        #         print("\n")


          def get_coherence(labels, token_lists, measure='c_v'):
              topics = get_topic_words(token_lists, labels)
              cm = CoherenceModel(topics=topics, texts=token_lists, corpus=corpus, dictionary=dictionary,coherence=measure)
              return cm.get_coherence()
          
          topic_words = []
          for i in range(0, len(top_n_words)-1):
            topic = []
            for tupl in top_n_words[i]:
              topic.append(tupl[0])
            topic_words.append(topic)
          def topic_diversity(topic_words):
              topk = 10
              if topic_words is None:
                  return 0
              if topk > len(topic_words[0]):
                  raise Exception('Words in topics are less than ' + str(topk))
              else:
                  unique_words = set()
                  for topic in topic_words:
                      unique_words = unique_words.union(set(topic[:topk]))
                  td = len(unique_words) / (topk * len(topic_words))
                  return td    
          
          with open("x_rus", "rb") as fp:   # Unpickling
             x_rus = pickle.load(fp)
        #   print('_________________________________________________________________')
        #   print('C_v coherence: ', get_coherence(list(docs_df['Topic']), x_rus, measure='c_v'))
        #   print('U_mass coherence: ', get_coherence(list(docs_df['Topic']), x_rus, measure='u_mass'))
        #   print('UCI : ', get_coherence(list(docs_df['Topic']), x_rus, measure='c_uci'))
        #   print('NPMI : ', get_coherence(list(docs_df['Topic']), x_rus, measure='c_npmi'))
        #   print('Topic_diversity : ', topic_diversity(topic_words))
        #   print('_________________________________________________________________')
        #   print(top_n_words)
          return top_n_words



