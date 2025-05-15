# === SAGE14-FX v2.1: Emotionally Stable Edition ===
# EpisodicMemory improved for batch dimension.

import tensorflow as tf

class EpisodicMemory(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.buffer = None

    def reset(self):
        self.buffer = None

    def write(self, embedding):
        if self.buffer is None:
            self.buffer = embedding[tf.newaxis, ...]  # (1, B, D)
        else:
            self.buffer = tf.concat([self.buffer, embedding[tf.newaxis, ...]], axis=0)

    def read_all(self):
        if self.buffer is None:
            return tf.zeros((1, 1, 1))
        return self.buffer  # (T, B, D)

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
    def __init__(self, dim, input_dim):
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


class Sage14FX(tf.keras.Model):
    def __init__(self, hidden_dim, shots=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shots = shots
        self.encoder = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation='relu')
        self.norm = tf.keras.layers.BatchNormalization()
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.pain_system = TaskPainSystem(hidden_dim)

        # input_dim = task_embed (D) + memory_context (T*D)
        input_dim = hidden_dim * (1 + shots)
        self.chooser = ChoiceHypothesisModule(hidden_dim, input_dim=input_dim)

        self.decoder = tf.keras.layers.Conv2D(10, (1, 1))
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
            x = self.norm(x, training=training)
            x_flat = tf.reduce_mean(x, axis=[1, 2])
            out, [state] = self.agent(x_flat, [state])
            self.memory.write(out)

        task_embed = state  # (B, D)
        memory_tensor = self.memory.read_all()  # (T, B, D)
        memory_tensor = tf.transpose(memory_tensor, [1, 0, 2])  # (B, T, D)
        memory_context = tf.reshape(memory_tensor, [batch, T * self.hidden_dim])  # (B, T*D)

        full_context = tf.concat([task_embed, memory_context], axis=-1)  # (B, D + T*D)
        full_context = tf.reshape(full_context, [batch, 1, 1, -1])
        full_context = tf.tile(full_context, [1, 20, 20, 1])

        chosen_transform = self.chooser(full_context)
        output_logits = self.decoder(chosen_transform)

        if y_seq is not None:
            last_y = tf.one_hot(y_seq[:, -1], depth=10, dtype=tf.float32)
            pain, gate = self.pain_system(output_logits, last_y)
            self._pain = pain
            self._gate = gate
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            self._loss_pain = loss_fn(last_y, output_logits)

        return output_logits
