import json

from tensorflow.keras.models import load_model
import tensorflow as tf

model_path = '../models/ko_spacing'
model = load_model(model_path)

training_config = '../resources/config.json'
char_file = '../resources/chars-4996'

with open(training_config, encoding='utf-8') as f:
    config = json.load(f)

with open(char_file, encoding='utf=8') as f:
    content = f.read()
    keys = ["<pad>", "<s>", "</s>", "<unk>"] + list(content)
    values = list(range(len(keys)))

    vocab_initializer = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int32)
    vocab_table = tf.lookup.StaticHashTable(vocab_initializer, default_value=3)

def predict(model, vocab_table, input_str):
    inference = get_inference_fn(model, vocab_table)
    input_str = tf.constant(input_str)
    result = inference(input_str).numpy()
    #print(b"".join(result).decode("utf8"))
    return b"".join(result).decode("utf8")

def get_inference_fn(model, vocab_table):
    @tf.function
    def inference(tensors):
        byte_array = tf.concat(
            [["<s>"], tf.strings.unicode_split(tf.strings.regex_replace(tensors, " +", " "), "UTF-8"), ["</s>"]], axis=0
        )
        strings = vocab_table.lookup(byte_array)[tf.newaxis, :]

        model_output = tf.argmax(model(strings), axis=-1)[0]
        return convert_output_to_string(byte_array, model_output)

    return inference

def convert_output_to_string(byte_array, model_output):
    sequence_length = tf.size(model_output)
    while_condition = lambda i, *_: i < sequence_length

    def while_body(i, o):
        o = tf.cond(
            model_output[i] == 1,
            lambda: tf.concat([o, [byte_array[i], " "]], axis=0),
            lambda: tf.cond(
                (model_output[i] == 2) and (byte_array[i] == " "),
                lambda: o,
                lambda: tf.concat([o, [byte_array[i]]], axis=0),
                ),
            )
        return i + 1, o

    _, strings_result = tf.while_loop(
        while_condition,
        while_body,
        (tf.constant(0), tf.constant([], dtype=tf.string)),
        shape_invariants=(tf.TensorShape([]), tf.TensorShape([None])),
    )
    return strings_result

input_str = '오늘은우울한날이야...ㅜㅜ'
result = predict(model, vocab_table, input_str)
