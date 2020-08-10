import pandas as pd


class CovidHelpers(object):
    def __init__(self, assets_path):
        self.meta_container = {}
        self.summary_container = {}
        self.literature_container = {}
        self.assets_path = assets_path

    def populate_meta_frame(self):
        meta_path = self.assets_path + '/metadata.csv'
        _data = pd.read_csv(meta_path)
        return _data

    def get_text_dataframe(self, file_name):
      data_path = self.assets_path + '/' + file_name
      _data = pd.read_csv(data_path)

      return _data

    def get_dt_info(self, dt):
      return dt.info()

    def get_topics(self, lda_model, count_vector, num_limit):
      feature_names = count_vector.get_feature_names()
      for index, topic in enumerate(lda_model.components_):
        print("Topic Number: ", index)
        print("\n")
        tpcs = " ".join([feature_names[itm]
                         for itm in topic.argsort()[:-num_limit - 1:-1]])
        print("Topic: " tpcs)
