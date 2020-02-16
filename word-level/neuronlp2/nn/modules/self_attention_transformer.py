import torch

class TransformerSelfAttention(torch.nn.Module):
    def __init__(self,
                 num_heads,
                 model_dim,
                 dropout_keep_prob):
        super(TransformerSelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout_keep_prob = dropout_keep_prob
        self.q_layer = torch.nn.Linear(model_dim, model_dim * self.num_heads, bias=False)
        self.k_layer = torch.nn.Linear(model_dim, model_dim * self.num_heads, bias=False)
        self.v_layer = torch.nn.Linear(model_dim, model_dim * self.num_heads, bias=False)
        self.out_layer = torch.nn.Linear(model_dim * self.num_heads, model_dim, bias=False)
        self.layer_norm1 = torch.nn.LayerNorm(model_dim)
        self.layer_norm2 = torch.nn.LayerNorm(model_dim)
        self.ff1 = torch.nn.Linear(model_dim, model_dim * 4, bias=True)
        self.ff2 = torch.nn.Linear(model_dim * 4, model_dim, bias=True)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(1- dropout_keep_prob)
        

    def forward(self, batched_inputs, attn_mask=None):
        q, k, v = self._linear_projection(batched_inputs)
        qs, ks, vs = self._split_heads(q, k, v)
        
        outputs = self._scaled_dot_product(qs, ks, vs, attn_mask)  # (batch, num_heads, max_contexts, value_dim)
        outputs = self._concat_heads(outputs)  # (batch, max_contexts, value_dim * num_heads)
        outputs = self.out_layer(outputs)  # (batch, max_contexts, model_dim)
        #outputs = self.relu(outputs)  # (batch, max_contexts, model_dim)
        outputs = self.dropout(outputs)
        
        attended = self.layer_norm1(outputs + batched_inputs)
        #outputs = torch.cat([outputs, batched_inputs], dim=-1) # (batch, max_contexts, 2 * model_dim)
        ff1 = self.ff1(attended)
        ff1 = self.relu(ff1)  # (batch, max_contexts, model_dim)
        ff2 = self.ff2(ff1)
        ff2 = self.dropout(ff2)
        
        result = self.layer_norm2(ff2 + attended)
        return result

    def _linear_projection(self, batched_inputs):
        q = self.q_layer(batched_inputs)  # (batch, max_contexts, key_dim * num_heads)
        k = self.q_layer(batched_inputs)  # (batch, max_contexts, key_dim * num_heads)
        v = self.q_layer(batched_inputs)  # (batch, max_contexts, key_dim * num_heads)
        # k = tf.layers.dense(batched_inputs, units=self.model_dim,
        #                     use_bias=False)  # (batch, max_contexts, key_dim * num_heads)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            tensor = tensor.view([-1, tensor.size()[1], num_heads,
                                         dim])  # (batch, max_contexts, num_heads, dim)
            return tensor.transpose(1,2)  # (batch, num_heads, max_contexts, dim)

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.model_dim)  # (batch, num_heads, max_contexts, key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.model_dim)  # (batch, num_heads, max_contexts, key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.model_dim)  # (batch, num_heads, max_contexts, key_dim)
        # ks = split_last_dimension_then_transpose(k, self.num_heads,
        #                                          self.model_dim)  # (batch, num_heads, max_contexts, key_dim)
        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs, valid_mask):
        queries_dot_keys = torch.matmul(qs, ks.transpose(2,3))  # (batch, num_heads, max_contexts, max_contexts)
        scaled_scores = queries_dot_keys / ((self.model_dim // self.num_heads) ** 0.5)  # (batch, num_heads, max_contexts, max_contexts)

        if valid_mask is not None:
            mask = torch.log(valid_mask.view(valid_mask.size()[0], 1, 1, valid_mask.size()[1])) # (batch, 1, 1, max_contexts)
            scaled_scores += mask

        attention_weights = self.softmax(scaled_scores)  # (batch, num_heads, max_contexts, max_contexts)
        return torch.matmul(attention_weights, vs)  # (batch, num_heads, max_contexts, value_dim)

    def _concat_heads(self, outputs):
        # outputs: (batch, num_heads, max_contexts, value_dim)
        max_contexts = outputs.size()[2]
        tensor = outputs.transpose(1, 2)  # [batch, max_contexts, num_heads, value_dim]
        return tensor.contiguous().view([-1, max_contexts, self.model_dim * self.num_heads])
