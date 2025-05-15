# === SAGE14-FX v3.0: Emotionally Flexible Edition ===
# EpisodicMemory handles variable shots, dynamic channels respected.

import tensorflow as tf

class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def reset(self):
        self.buffer = []

    def write(self, embedding):
        self.buffer.append(embedding)

    def read_all(self):
        if not self.buffer:
            return tf.zeros((1, 1, 1))
        return tf.stack(self.buffer, axis=0)

class TaskPainSystem(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.threshold = tf.Variable(0.1, trainable=True)
        self.sensitivity = tf.Variable(tf.ones([1, 1, 1, 10]), trainable=True)

    def call(self, pred, expected):
        diff = tf.square(pred - expected)
        pain = tf.reduce_mean(self.sensitivity * diff)
        gate = tf.sigmoid((pain - self.threshold) * 10.0)
        return pain, gate

class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu')
        self.hypotheses = [tf.keras.layers.Conv2D(dim, kernel_size=1, activation='relu') for _ in range(4)]
        self.selector = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x):
        x = self.input_proj(x)
        candidates = [h(x) for h in self.hypotheses]
        stacked = tf.stack(candidates, axis=1)
        weights = self.selector(tf.reduce_mean(x, axis=[1, 2]))
        weights = tf.reshape(weights, [-1, 4, 1, 1, 1])
        chosen = tf.reduce_sum(stacked * weights, axis=1)
        return chosen

class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
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

class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query = tf.keras.layers.Conv2D(dim, 1)
        self.key = tf.keras.layers.Conv2D(dim, 1)
        self.value = tf.keras.layers.Conv2D(dim, 1)
        self.out = tf.keras.layers.Conv2D(dim, 1)

    def call(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_logits = tf.reduce_sum(q * k, axis=-1, keepdims=True)
        attn = tf.nn.softmax(tf.reshape(attn_logits, [tf.shape(x)[0], -1]), axis=-1)
        attn = tf.reshape(attn, tf.shape(attn_logits))
        out = v * attn
        return self.out(out)

class Sage14FX(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        kernel_init = tf.keras.initializers.HeNormal()

        self.hidden_dim = hidden_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init),
            tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init),
        ])
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.attn = SimpleAttention(hidden_dim + 2)
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.pain_system = TaskPainSystem(hidden_dim)
        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(10, (1, 1), kernel_initializer=kernel_init)
        ])
        self._pain = None
        self._gate = None
        self._loss_pain = None

    def call(self, x_seq, y_seq=None, training=False):
        batch = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]
        state = tf.zeros([batch, self.hidden_dim])
        self.memory.reset()

        for t in range(T):
            x = x_seq[:, t]
            x = self.encoder(x)
            x = self.norm(x)
            x_flat = tf.reduce_mean(x, axis=[1, 2])
            out, [state] = self.agent(x_flat, [state])
            self.memory.write(out)

        task_embed = state
        memory_tensor = self.memory.read_all()
        memory_tensor = tf.transpose(memory_tensor, [1, 0, 2])
        memory_context = tf.reshape(memory_tensor, [batch, -1])

        full_context = tf.concat([task_embed, memory_context], axis=-1)
        full_context = tf.reshape(full_context, [batch, 1, 1, -1])
        full_context = tf.tile(full_context, [1, 20, 20, 1])

        full_context = self.pos_enc(full_context)
        full_context = self.attn(full_context)

        chosen_transform = self.chooser(full_context)

        output_logits = self.decoder(chosen_transform)

        if y_seq is not None:
            expected = tf.one_hot(y_seq[:, -1], depth=10, dtype=tf.float32)
            pain, gate = self.pain_system(output_logits, expected)
            self._pain = pain
            self._gate = gate
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self._loss_pain = loss_fn(y_seq[:, -1], output_logits)

        return output_logits
