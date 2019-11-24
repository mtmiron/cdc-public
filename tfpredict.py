import tensorflow as tf
import pandas as pd
import numpy as np
import transformers
import plac
import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, load_model=False):
        self.config = transformers.DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=100)
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = transformers.TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=self.config)

    def get_dataset(self, df, labels=False):
        #dataset: {idx: (), label: (), sentence1: (), sentence2: ()}, types: {idx: tf.int32, label: tf.int64, sentence1: tf.string, sentence2: tf.string}>
        idxs = []
        labels = []
        sentences1 = []
        sentences2 = []
        for i, ex in enumerate(df['text']):
            idx = i
            if labels:
                label = df['event'][i]
            else:
                label = 0
            sentence1 = ex
            sentence2 = ""
            idxs.append(idx)
            labels.append(label)
            sentences1.append(sentence1)
            sentences2.append(sentence2)
        ds = tf.data.Dataset.from_tensor_slices({'idx': idxs, 'label': labels, 'sentence1': sentences1, 'sentence2': sentences2})
        features = transformers.glue_convert_examples_to_features(ds, self.tokenizer, 64, task='mrpc', output_mode='classification', label_list=[str(x) for x in range(100)])
        return features

    @staticmethod
    def load_model(path="checkpoints/tf/bert/model.tf"):
        model = Model()
        model.model.load_weights(path)
        return model
        self = Model()
        self.model = model
        return self

    def predict_df(self, df):
        """"
        tokens = []
        for text in df['text']:
            enc = self.tokenizer.encode_plus(text, add_special_tokens=True)
            tokens.append(enc)
        """
        ds = self.get_dataset(df)
        return self.predict_ds(ds)
        return np.argmax(ret, axis=1), ret

    def predict_ds(self, dataset):
        dataset.batch(8)
        return self.model.predict(dataset)

    def predict_text(self, strings):
        df = pd.DataFrame({'text': strings})
        ds = self.get_dataset(df, labels=False)
        return self.predict_ds(ds)

def main(modelpath, datapath):
    import main
    logger.info("Loading model")
    model = Model.load_model(modelpath)
    df = pd.read_csv(datapath)
    logger.info("Predicting")
    ret, _ = model.predict_df(df)
    import pdb; pdb.set_trace()
    solution = []
    for i in tqdm.tqdm(range(len(ret))):
        solution.append(ret[i]+1)
    with open(datapath) as f:
        gt = f.readlines()
    print("Score: {}".format(main.score(gt[1:], solution)))

if __name__ == "__main__":
    plac.call(main)
