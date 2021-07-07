import tensorflow as tf
import numpy as np
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.optimizers import Adam


class BETO:
    def __init__(self):
        self.diagnostics = {
            0:'CARDIOLOGIA',
            1:'CIRUGIA ADULTO',
            2:'DERMATOLOGIA',
            3:'ENDODONCIA',
            4:'GASTROENTEROLOGIA',
            5:'GINECOLOGIA',
            6:'MEDICINA INTERNA',
            7:'NEUROLOGIA',
            8:'OFTALMOLOGIA',
            9:'OTORRINOLARINGOLOGIA',
            10:'PEDIATRIA',
            11:'TRAUMATOLOGIA',
            12:'UROLOGIA',
        }
        self.bert_tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.bert_model = TFBertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

        self.model = self.create_model(self.bert_model)
        tf.keras.Model.load_weights(self.model, 'model.h5')

    def create_model(self, bert_model):
        input_ids = tf.keras.Input(shape=(32,),
                                   dtype='int32')
        attention_masks = tf.keras.Input(shape=(32,),
                                         dtype='int32')
        output = self.bert_model([input_ids, attention_masks])
        print(
            output)
        output = output[
            1]
        output = tf.keras.layers.Dense(128, activation='relu')(
            output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(13, activation='softmax')(output)
        model = tf.keras.models.Model(inputs=[input_ids, attention_masks],
                                      outputs=output)
        model.compile(Adam(lr=6e-6), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def infer(self, data):
        spec = []
        input_ids = []
        attention_masks = []
        encoded = self.bert_tokenizer.encode_plus(data, add_special_tokens=True, max_length=32, pad_to_max_length=True,
                                                  return_attention_mask=True, )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        pred = np.array(self.model.predict([np.array(input_ids), np.array(attention_masks)]))
        for i in range(13):
            spec.append((self.diagnostics[i], round(pred[0][i] * 100, 2)))
        return spec
