# === SAGE14-FX v2: Emotionally Mature Edition ===
# EpisodicMemory now handles variable-length few-shot episodes.

import tensorflow as tf

class EpisodicMemory(tf.keras.layers.Layer):
    """Flexible memory that accumulates task-specific embeddings across time."""
    def __init__(self):
        super().__init__()
        self.buffer = []

    def reset(self):
        self.buffer = []

    def write(self, embedding):
        self.buffer.append(embedding)

    def read_all(self):
        return tf.stack(self.buffer, axis=0) if self.buffer else tf.zeros((1, 1))

class TaskPainSystem(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.threshold = tf.Variable(0.1, trainable=True)
        self.sensitivity = tf.Variable(tf.ones([1, dim]), trainable=True)

    def call(self, pred, expected):
        diff = tf.square(pred - expected)
        pain = tf.reduce_mean(self.sensitivity * diff)
        gate = tf.sigmoid((pain - self.threshold) * 10.0)
        return pain, gate

class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.hypotheses = [tf.keras.layers.Dense(dim, activation='relu') for _ in range(4)]
        self.selector = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x):
        candidates = [h(x) for h in self.hypotheses]
        stacked = tf.stack(candidates, axis=1)
        weights = self.selector(tf.reduce_mean(x, axis=[1, 2]))
        chosen = tf.reduce_sum(stacked * tf.expand_dims(tf.expand_dims(weights, 2), 3), axis=1)
        return chosen

class Sage14FX(tf.keras.Model):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = tf.keras.layers.Conv2D(hidden_dim, (3, 3), padding='same', activation='relu')
        self.norm = tf.keras.layers.LayerNormalization()
        self.agent = tf.keras.layers.GRUCell(hidden_dim)
        self.memory = EpisodicMemory()
        self.pain_system = TaskPainSystem(hidden_dim)
        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.decoder = tf.keras.layers.Conv2D(10, (1, 1))

    def call(self, x_seq, y_seq=None, training=False):
        batch = tf.shape(x_seq)[0]
        T = tf.shape(x_seq)[1]
        state = tf.zeros([batch, self.agent.units])
        self.memory.reset()

        for t in range(T):
            x = x_seq[:, t]  # (B, H, W, C)
            x = self.encoder(x)
            x = self.norm(x)
            x_flat = tf.reduce_mean(x, axis=[1, 2])
            out, state = self.agent(x_flat, [state])
            self.memory.write(out)

        task_embed = state
        memory_context = tf.reduce_mean(self.memory.read_all(), axis=0, keepdims=True)
        full_context = tf.concat([task_embed, memory_context], axis=-1)
        full_context = tf.reshape(full_context, [batch, 1, 1, -1])
        full_context = tf.tile(full_context, [1, 20, 20, 1])

        chosen_transform = self.chooser(full_context)
        output_logits = self.decoder(chosen_transform)

        if y_seq is not None:
            last_y = tf.one_hot(y_seq[:, -1], depth=10, dtype=tf.float32)
            pain, gate = self.pain_system(output_logits, last_y)
            self.add_metric(pain, name="task_pain")
            self.add_metric(gate, name="adaptation_gate")

        return output_logits
