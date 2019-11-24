import tensorflow as tf
import transformers
import preproc
import pandas as pd
import numpy as np
import logging
import transformers
import tensorflow_datasets
import tfpredict
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BATCH_SIZE = EVAL_BATCH_SIZE = 8
USE_AMP = True
USE_XLA = False

SAVEDIR = "cp/tf/distilbert"
BASEMODEL = "distilbert-base-uncased"
MODELCLASSES = [ transformers.TFDistilBertForSequenceClassification,
                 transformers.DistilBertConfig,
                 transformers.DistilBertTokenizer ]

train_df = pd.read_csv("train5000.csv")
train_df = preproc.preproc(train_df, lower=True)
valid_df = pd.read_csv("valid5000.csv")
valid_df = preproc.preproc(valid_df)

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

config = MODELCLASSES[1].from_pretrained(BASEMODEL, num_labels=100)
model = MODELCLASSES[0].from_pretrained(BASEMODEL, config=config)
#model.load_weights("cp/tf/bert/model_weights.hdf5")
tokenizer = MODELCLASSES[2].from_pretrained(BASEMODEL)

def get_dataset(train_df):
    #dataset: {idx: (), label: (), sentence1: (), sentence2: ()}, types: {idx: tf.int32, label: tf.int64, sentence1: tf.string, sentence2: tf.string}>
    idxs = []
    labels = []
    sentences1 = []
    sentences2 = []
    for i, ex in enumerate(train_df['text']):
        idx = i
        label = train_df['event'][i]
        sentence1 = ex
        sentence2 = ""
        idxs.append(idx)
        labels.append(label)
        sentences1.append(sentence1)
        sentences2.append(sentence2)
    ds = tf.data.Dataset.from_tensor_slices({'idx': idxs, 'label': labels, 'sentence1': sentences1, 'sentence2': sentences2})
    features = transformers.glue_convert_examples_to_features(ds, tokenizer, 64, task='mrpc', output_mode='classification', label_list=[str(x) for x in range(100)])
    return features

class F1(tf.keras.callbacks.Callback):
    def f1(self, truth, predictions):
        r = tf.compat.v1.metrics.recall(truth, predictions)
        p = tf.compat.v1.metrics.precision(truth, predictions)
        return 2 * r * p / (r + p)

    def on_epoch_end(self, epoch, logs=None):
        y_true = np.array(list(valid_df['event']))
        ds = get_dataset(valid_df)
        y_pred = np.argmax(self.model.predict(ds.batch(8)), axis=1)
        return self.f1(y_true, y_pred)

train_data = get_dataset(train_df).batch(BATCH_SIZE).repeat()
#train_data = their_dataset()
valid_data = get_dataset(valid_df).batch(EVAL_BATCH_SIZE).repeat()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-6)
if USE_AMP:
    # loss scaling is currently required when using mixed precision
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

save_cb = tf.keras.callbacks.ModelCheckpoint(SAVEDIR, mode="max", save_best_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=SAVEDIR + "/logs")

train_steps = len(train_df) // BATCH_SIZE // 8
valid_steps = len(valid_df) // EVAL_BATCH_SIZE

try:
    history = model.fit(train_data, validation_data=valid_data,
                        verbose=1, epochs=100, steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        callbacks=[save_cb, reduce_lr, F1()])
except Exception as exc:
    print(str(exc))
finally:
    try:
        model.save_weights(SAVEDIR + "/model_weights.hdf5")
    except Exception:
        pass
    try:
        model.save(SAVEDIR + "/model.tf", save_format="tf")
    except Exception:
        pass
    model.save_pretrained(SAVEDIR)
