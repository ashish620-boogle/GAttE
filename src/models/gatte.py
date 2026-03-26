from __future__ import annotations

from typing import Dict

import tensorflow as tf
import tensorflow_hub as hub

try:
    from tensorflow.keras.layers import Conv1DTranspose
except Exception:  # pragma: no cover
    class Conv1DTranspose(tf.keras.layers.Layer):
        def __init__(self, filters, kernel_size, strides=1, padding="same", activation=None, **kwargs):
            super().__init__()
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=(kernel_size, 1),
                strides=(strides, 1),
                padding=padding,
                activation=activation,
                **kwargs,
            )

        def call(self, x):
            x = tf.expand_dims(x, axis=2)
            y = self.conv(x)
            return tf.squeeze(y, axis=2)


def _concat_or_single(tensors, name: str):
    active = [tensor for tensor in tensors if tensor is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return tf.keras.layers.Concatenate(name=name)(active)


def build_gatte(
    cfg: Dict,
    num_classes: int,
    word_vocab_size: int,
    char_vocab_size: int,
    max_char_len: int,
    use_precomputed: bool = False,
) -> tf.keras.Model:
    mcfg = cfg["model"]
    ppcfg = cfg["preprocess"]
    attention_mode = mcfg.get("attention_mode", "multihead")
    no_attention_mode = mcfg.get("no_attention_mode", "concat_qv")

    max_words = ppcfg["max_words"]

    if use_precomputed:
        use_input = tf.keras.Input(shape=(512,), dtype=tf.float32, name="use_vec")
        use_vec = use_input
    else:
        use_input = tf.keras.Input(shape=(), dtype=tf.string, name="use_text")
        use_layer = hub.KerasLayer(mcfg["use_url"], trainable=False, name="use")
        use_vec = use_layer(use_input)
    if not mcfg.get("use_sentence_embedding", True):
        use_vec = None

    char_input = tf.keras.Input(shape=(max_char_len,), dtype=tf.int32, name="char_input")
    char_flat = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name="char_float")(char_input)
    char_flat = tf.keras.layers.LayerNormalization(name="char_ln")(char_flat)
    if not mcfg.get("use_char_embedding", True):
        char_flat = None

    # Embedding ablations remove disabled branches entirely instead of injecting zeros.
    concat_xy = _concat_or_single([use_vec, char_flat], "concat_xy")
    kv = None
    if concat_xy is not None:
        ff1 = tf.keras.layers.Dense(mcfg["dense_units"], activation="relu", name="ff1")(concat_xy)
        if mcfg["dropout_rates"]:
            ff1 = tf.keras.layers.Dropout(mcfg["dropout_rates"][0], name="dropout_ff1")(ff1)

        timesteps = mcfg["timesteps"]
        channels = mcfg["dense_units"] // timesteps
        if mcfg["dense_units"] % timesteps != 0:
            raise ValueError("dense_units must be divisible by timesteps")
        reshaped = tf.keras.layers.Reshape((timesteps, channels), name="reshape_ff1")(ff1)

        if mcfg["use_deconv"]:
            kv = Conv1DTranspose(
                filters=mcfg["deconv_filters"][0],
                kernel_size=3,
                padding="same",
                activation="relu",
                name="deconv_kv_1",
            )(reshaped)
            kv = tf.keras.layers.LayerNormalization(name="ln_kv")(kv)
            kv = Conv1DTranspose(
                filters=mcfg["deconv_filters"][1],
                kernel_size=3,
                padding="same",
                activation="relu",
                name="deconv_kv_2",
            )(kv)
        else:
            kv = reshaped

    word_input = tf.keras.Input(shape=(max_words,), dtype=tf.int32, name="word_input")
    q = None
    if mcfg.get("use_word_embedding", True):
        word_emb = tf.keras.layers.Embedding(
            word_vocab_size,
            mcfg["word_dim"],
            name="word_embedding",
        )(word_input)

        if mcfg["use_deconv"]:
            q = Conv1DTranspose(
                filters=mcfg["q_filters"],
                kernel_size=3,
                padding="same",
                activation="relu",
                name="deconv_q",
            )(word_emb)
        else:
            q = word_emb

    if q is not None and kv is not None and (not mcfg.get("use_attention", True) or attention_mode == "none"):
        gb_k = tf.keras.layers.GlobalAveragePooling1D(name="gap_k")(kv)
        gb_v = tf.keras.layers.GlobalAveragePooling1D(name="gap_v")(kv)
        gb_q = tf.keras.layers.GlobalAveragePooling1D(name="gap_q")(q)
        if no_attention_mode == "concat_seq_gap":
            x1 = tf.keras.layers.Concatenate(name="concat_seq")([q, kv])
            x1 = tf.keras.layers.GlobalAveragePooling1D(name="gap_concat_seq")(x1)
        else:
            if no_attention_mode == "value_only_keep_q":
                # Keep the query branch connected so the ablation preserves the model size,
                # while removing its predictive contribution from the classifier head.
                gb_q = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x), name="gap_q_zero")(gb_q)
                x1 = tf.keras.layers.Concatenate(name="concat_gb")([gb_k, gb_q])
            elif no_attention_mode == "concat_qkv":
                x1 = tf.keras.layers.Concatenate(name="concat_gb")([gb_q, gb_k, gb_v])
            else:
                x1 = tf.keras.layers.Concatenate(name="concat_gb")([gb_k, gb_q])
    elif q is not None and kv is not None and attention_mode in {"simple", "self"}:
        attn_name = "self_attention" if attention_mode == "self" else "simple_attention"
        attn = tf.keras.layers.Attention(
            use_scale=False,
            score_mode="dot",
            name=attn_name,
        )([q, kv, kv])
        gb1 = tf.keras.layers.GlobalAveragePooling1D(name="gap_attn")(attn)
        gb2 = tf.keras.layers.GlobalAveragePooling1D(name="gap_q")(q)
        x1 = tf.keras.layers.Concatenate(name="concat_gb")([gb2, gb1])
    elif q is not None and kv is not None:
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=mcfg["num_heads"],
            key_dim=mcfg["key_dim"],
            value_dim=mcfg["value_dim"],
            name="mha",
        )(query=q, key=kv, value=kv)
        gb1 = tf.keras.layers.GlobalAveragePooling1D(name="gap_attn")(attn)
        gb2 = tf.keras.layers.GlobalAveragePooling1D(name="gap_q")(q)
        x1 = tf.keras.layers.Concatenate(name="concat_gb")([gb2, gb1])
    elif q is not None:
        x1 = tf.keras.layers.GlobalAveragePooling1D(name="gap_q_only")(q)
    elif kv is not None:
        x1 = tf.keras.layers.GlobalAveragePooling1D(name="gap_kv_only")(kv)
    else:
        raise ValueError("At least one embedding branch must be enabled.")
    inputs = [use_input, word_input, char_input]

    variant = mcfg.get("variant", "paper")
    if variant == "colab_v4":
        x = tf.keras.layers.LayerNormalization(name="ln_head")(x1)
        x = tf.keras.layers.Dense(256, activation=None, name="head_dense_256")(x)
        if mcfg.get("use_activity_regularizer", False):
            x = tf.keras.layers.ActivityRegularization(
                l1=mcfg.get("activity_l1", 0.0), l2=mcfg.get("activity_l2", 0.0), name="head_actreg_1"
            )(x)
        x = tf.keras.layers.Dense(num_classes * 2, activation=None, name="head_dense_2c")(x)
        if mcfg.get("use_activity_regularizer", False):
            x = tf.keras.layers.ActivityRegularization(
                l1=mcfg.get("activity_l1", 0.0), l2=mcfg.get("activity_l2", 0.0), name="head_actreg_2"
            )(x)
        if len(mcfg["dropout_rates"]) > 1:
            x = tf.keras.layers.Dropout(mcfg["dropout_rates"][1], name="dropout_head")(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    else:
        ff2 = tf.keras.layers.Dense(mcfg["dense_units"], activation="relu", name="ff2")(x1)
        if mcfg.get("use_activity_regularizer", False):
            ff2 = tf.keras.layers.ActivityRegularization(
                l1=mcfg.get("activity_l1", mcfg.get("l1", 0.0)),
                l2=mcfg.get("activity_l2", mcfg.get("l2", 0.0)),
                name="ff2_actreg",
            )(ff2)
        if len(mcfg["dropout_rates"]) > 1:
            ff2 = tf.keras.layers.Dropout(mcfg["dropout_rates"][1], name="dropout_ff2")(ff2)
        out = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(ff2)

    model = tf.keras.Model(inputs=inputs, outputs=out, name="GAttE")
    return model
