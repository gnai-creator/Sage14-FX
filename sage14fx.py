# === SAGE14-FX v4.3: Pain-Aware Curious Hydra ===

import tensorflow as tf

class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.buffer = None

    def reset(self):
        self.buffer = None

    def write(self, embedding):
        if self.buffer is None:
            self.buffer = embedding[tf.newaxis, ...]
        else:
            self.buffer = tf.concat([self.buffer, embedding[tf.newaxis, ...]], axis=0)

    def read_all(self):
        if self.buffer is None:
            return tf.zeros((1, 1, 1))
        return self.buffer


class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        y_pos = tf.linspace(-1.0, 1.0, tf.cast(h, tf.int32))
        x_pos = tf.linspace(-1.0, 1.0, tf.cast(w, tf.int32))
        yy, xx = tf.meshgrid(y_pos, x_pos, indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [b, 1, 1, 1])
        pos = self.dense(pos)
        return tf.concat([x, pos], axis=-1)


class FractalEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.branch3 = tf.keras.layers.Conv2D(dim // 2, kernel_size=3, padding='same', activation='relu')
        self.branch5 = tf.keras.layers.Conv2D(dim // 2, kernel_size=5, padding='same', activation='relu')
        self.merge = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same', activation='relu')
        self.residual = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same')  # For identity path

    def call(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        merged = tf.concat([b3, b5], axis=-1)
        out = self.merge(merged)
        skip = self.residual(x)
        return tf.nn.relu(out + skip)



class FractalBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(dim, kernel_size=3, padding='same', activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.skip = tf.keras.layers.Conv2D(dim, kernel_size=1, padding='same')

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)
        skip = self.skip(x)
        return tf.nn.relu(out + skip)



class MultiHeadAttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)

    def call(self, x):
        return self.attn(query=x, value=x, key=x)


class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu')
        self.hypotheses = [tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu') for _ in range(4)]
        self.selector = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x, hard=False):
        x = self.input_proj(x)
        candidates = [h(x) for h in self.hypotheses]
        stacked = tf.stack(candidates, axis=1)
        weights = self.selector(tf.reduce_mean(x, axis=[1, 2]))

        if hard:
            idx = tf.argmax(weights, axis=-1)  # [B]
            one_hot = tf.one_hot(idx, depth=4, dtype=tf.float32)[:, :, tf.newaxis, tf.newaxis, tf.newaxis]
            return tf.reduce_sum(stacked * one_hot, axis=1)
        else:
            weights = tf.reshape(weights, [-1, 4, 1, 1, 1])
            return tf.reduce_sum(stacked * weights, axis=1)


class TaskPainSystem(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.threshold = tf.Variable(0.1, trainable=True)
        self.sensitivity = tf.Variable(tf.ones([1, 1, 1, 10]), trainable=True)

    def call(self, pred, expected):
        diff = tf.square(pred - expected)
        raw_pain = tf.reduce_mean(self.sensitivity * diff)
        exploration_gate = tf.clip_by_value(tf.nn.sigmoid((raw_pain - 2.0) * 0.5), 0.0, 1.0)
        adjusted_pain = raw_pain * (1.0 - exploration_gate)  # Fury mode: high exploration = less perceived pain
        gate = tf.sigmoid((adjusted_pain - self.threshold) * 10.0)

        tf.print("Pain:", raw_pain, "Fury_Pain:" adjusted_pain, "Gate:", gate, "Exploration Gate:", exploration_gate)
        return pain, gate


class AttentionOverMemory(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query_proj = tf.keras.layers.Dense(dim)
        self.key_proj = tf.keras.layers.Dense(dim)
        self.value_proj = tf.keras.layers.Dense(dim)

    def call(self, memory, query):
        q = self.query_proj(query)[:, tf.newaxis, :]  # [B, 1, D]
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        attn_weights = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True), axis=1)
        attended = tf.reduce_sum(attn_weights * v, axis=1)
        return attended


class Sage14FX(tf.keras.Model):
    def __init__(self, hidden_dim, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice
        self.encoder = tf.keras.Sequential([
            FractalEncoder(hidden_dim),
            FractalBlock(hidden_dim),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu')
        ])
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(2)
        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.pain_system = TaskPainSystem(hidden_dim)
        self.attend_memory = AttentionOverMemory(hidden_dim)
        self.projector = tf.keras.layers.Conv2D(self.hidden_dim, 1)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(10, 1)
        ])
        self.gate_scale = tf.keras.layers.Dense(self.hidden_dim, activation='sigmoid')


    def call(self, x_seq, y_seq=None, training=False):
        batch = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]
        state = tf.zeros([batch, self.hidden_dim])
        self.memory.reset()

        for t in range(T):
            x = self.norm(self.encoder(x_seq[:, t]))
            x_flat = tf.reduce_mean(x, axis=[1, 2])
            out, [state] = self.agent(x_flat, [state])
            self.memory.write(out)

        memory_tensor = tf.transpose(self.memory.read_all(), [1, 0, 2])
        memory_context = self.attend_memory(memory_tensor, state)
        full_context = tf.concat([state, memory_context], axis=-1)
        context = tf.tile(tf.reshape(full_context, [batch, 1, 1, -1]), [1, 20, 20, 1])

        projected_input = self.projector(self.pos_enc(context))
        attended = self.attn(projected_input)
        chosen_transform = self.chooser(attended, hard=self.use_hard_choice)

        last_input_encoded = self.encoder(x_seq[:, -1])
        # Calcular o gate por canal com base no contexto de memória
        context_features = tf.concat([state, memory_context], axis=-1)
        channel_gate = self.gate_scale(context_features)
        channel_gate = tf.reshape(channel_gate, [batch, 1, 1, self.hidden_dim])
        channel_gate = tf.clip_by_value(channel_gate, 0.0, 1.0)
        blended = channel_gate * chosen_transform + (1 - channel_gate) * last_input_encoded
        merged = blended  # ou concat com outro tensor, se quiser
        output_logits = self.decoder(merged)

        if y_seq is not None:
            expected = tf.one_hot(y_seq[:, -1], depth=10, dtype=tf.float32)
            pain, gate = self.pain_system(output_logits, expected)
            self._pain = pain
            self._gate = gate
            self._loss_pain = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_seq[:, -1], output_logits)

        return output_logits
