import numpy as np
import torch
import time
import os
from transformers import AutoTokenizer
from huggingface_hub import login
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

class Model:
    def __init__(self, gguf_path: str, max_ctx_len: int = 8192, dtype: str = "float16", device: str = "cuda"):
        self.gguf_path = gguf_path
        self.reader = GGUFReader(gguf_path, "r")

        self.device = device
        self.dtype = dtype

        if self.dtype == "float16":
            self.numpy_dtype = np.float16
            self.torch_dtype = torch.float16
        elif self.dtype == "float32":
            self.numpy_dtype = np.float32
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}. Supported dtypes are 'float16' and 'float32'.")

        self.n_block      = self.reader.get_field("llama.block_count"                     ).parts[-1][0]
        # self.max_ctx_len  = self.reader.get_field("llama.context_length"                  ).parts[-1][0] # in case of CUDA out of memory
        self.d            = self.reader.get_field("llama.embedding_length"                ).parts[-1][0]
        self.d_ffn        = self.reader.get_field("llama.feed_forward_length"             ).parts[-1][0]
        self.n_head_q     = self.reader.get_field("llama.attention.head_count"            ).parts[-1][0]
        self.n_head_kv    = self.reader.get_field("llama.attention.head_count_kv"         ).parts[-1][0]
        self.freq_base    = self.reader.get_field("llama.rope.freq_base"                  ).parts[-1][0]
        self.rms_epsilon  = self.reader.get_field("llama.attention.layer_norm_rms_epsilon").parts[-1][0]
        self.d_k          = self.reader.get_field("llama.attention.key_length"            ).parts[-1][0]
        self.d_v          = self.reader.get_field("llama.attention.value_length"          ).parts[-1][0]
        self.n_vocab      = self.reader.get_field("llama.vocab_size"                      ).parts[-1][0]
        self.d_rope       = self.reader.get_field("llama.rope.dimension_count"            ).parts[-1][0]

        self.bos_token_id = self.reader.get_field("tokenizer.ggml.bos_token_id"           ).parts[-1][0]
        self.eos_token_id = self.reader.get_field("tokenizer.ggml.eos_token_id"           ).parts[-1][0]
        self.pad_token_id = self.reader.get_field("tokenizer.ggml.padding_token_id"       ).parts[-1][0]

        self.max_ctx_len = 8192 # in case CUDA out of memory

        print("Model Configuration:")
        print(f"    Device: {self.device}")
        print(f"    GGUF Path: {self.gguf_path}")
        print(f"    Vocabulary Size: {self.n_vocab}")
        print(f"    Maximum Context Length: {self.max_ctx_len}")
        print(f"    Transformer Block Count: {self.n_block}")
        print(f"    Embedding Dimension: {self.d}")
        print(f"    Attention Query Head Count: {self.n_head_q}")
        print(f"    Attention Key/Value Head Count: {self.n_head_kv}")
        print(f"    Attention Key Dimension: {self.d_k}")
        print(f"    Attention Value Dimension: {self.d_v}")
        print(f"    Feed Forward Dimension: {self.d_ffn}")
        print(f"    RoPE Base Frequency: {self.freq_base}")
        print(f"    RoPE Dimension: {self.d_rope}")
        print(f"    RMSNorm Epsilon: {self.rms_epsilon}")
        print(f"    BOS Token ID: {self.bos_token_id}")
        print(f"    EOS Token ID: {self.eos_token_id}")
        print(f"    Padding Token ID: {self.pad_token_id}")

        self.token_embed_weight = torch.empty(self.n_vocab, self.d, dtype=self.torch_dtype, device=device)                           # [n_vocab, d]

        self.attn_q_weight = torch.empty(self.n_block, self.n_head_q, self.d_k, self.d, dtype=self.torch_dtype, device=device)       # [n_block, n_head_q, d_k, d]
        self.attn_k_weight = torch.empty(self.n_block, self.n_head_kv, self.d_k, self.d, dtype=self.torch_dtype, device=device)      # [n_block, n_head_kv, d_k, d]
        self.attn_v_weight = torch.empty(self.n_block, self.n_head_kv, self.d_v, self.d, dtype=self.torch_dtype, device=device)      # [n_block, n_head_kv, d_v, d]
        self.attn_o_weight = torch.empty(self.n_block, self.d, self.n_head_q * self.d_v, dtype=self.torch_dtype, device=device)      # [n_block, d, n_head_q * d_v]
        self.attn_norm_weight = torch.empty(self.n_block, self.d, dtype=torch.float32, device=device)                                # [n_block, d]

        self.ffn_gate_weight = torch.empty(self.n_block, self.d_ffn, self.d, dtype=self.torch_dtype, device=device)                  # [n_block, d_ffn, d]
        self.ffn_up_weight   = torch.empty(self.n_block, self.d_ffn, self.d, dtype=self.torch_dtype, device=device)                  # [n_block, d_ffn, d]
        self.ffn_down_weight = torch.empty(self.n_block, self.d, self.d_ffn, dtype=self.torch_dtype, device=device)                  # [n_block, d, d_ffn]
        self.ffn_norm_weight = torch.empty(self.n_block, self.d, dtype=torch.float32, device=device)                                 # [n_block, d]

        self.rope_freqs_weight = torch.empty(self.d_rope // 2, dtype=torch.float32, device=device)                                   # [d_rope/2]
        self.rope_theta = torch.empty(self.d_rope // 2, dtype=torch.float32, device=device)                                          # [d_rope/2]

        self.output_norm_weight = torch.empty(self.d, dtype=torch.float32, device=device)                                            # [d]

        self.load_tensors()

        self.tokens = torch.empty((self.max_ctx_len,), dtype=torch.int64, device=device)                                             # [max_ctx_len]

        self.kcache = torch.empty((self.n_block, self.n_head_kv, self.max_ctx_len, self.d_k), dtype=self.torch_dtype, device=device) # [n_block, n_head_kv, max_ctx_len, d_k]
        self.vcache = torch.empty((self.n_block, self.n_head_kv, self.max_ctx_len, self.d_v), dtype=self.torch_dtype, device=device) # [n_block, n_head_kv, max_ctx_len, d_v]

        self.ctx_len = 0

    def initialize(self):
        self.ctx_len = 0

    def get_tensor_by_name(self, name: str):
        for tensor in self.reader.tensors:
            if tensor.name == name:
                return tensor
        raise ValueError(f"Tensor '{name}' not found in GGUF file.")

    def load_tensors(self):
        # Load token embedding weights
        token_embed_tensor = self.get_tensor_by_name("token_embd.weight")
        self.token_embed_weight.copy_(torch.from_numpy(dequantize(
            token_embed_tensor.data.copy(), token_embed_tensor.tensor_type).astype(self.numpy_dtype)).view(self.n_vocab, self.d))       # [n_vocab, d]

        # Load attention weights
        for i in range(self.n_block):
            attn_q_tensor = self.get_tensor_by_name(f"blk.{i}.attn_q.weight")
            self.attn_q_weight[i] = torch.from_numpy(dequantize(
                attn_q_tensor.data.copy(), attn_q_tensor.tensor_type).astype(self.numpy_dtype)).view(self.n_head_q, self.d_k, self.d)   # [n_head_q, d_k, d]

            attn_k_tensor = self.get_tensor_by_name(f"blk.{i}.attn_k.weight")
            self.attn_k_weight[i] = torch.from_numpy(dequantize(
                attn_k_tensor.data.copy(), attn_k_tensor.tensor_type).astype(self.numpy_dtype)).view(self.n_head_kv, self.d_k, self.d)  # [n_head_kv, d_k, d]

            attn_v_tensor = self.get_tensor_by_name(f"blk.{i}.attn_v.weight")
            self.attn_v_weight[i] = torch.from_numpy(dequantize(
                attn_v_tensor.data.copy(), attn_v_tensor.tensor_type).astype(self.numpy_dtype)).view(self.n_head_kv, self.d_v, self.d)  # [n_head_kv, d_v, d]

            attn_o_tensor = self.get_tensor_by_name(f"blk.{i}.attn_output.weight")
            self.attn_o_weight[i] = torch.from_numpy(dequantize(
                attn_o_tensor.data.copy(), attn_o_tensor.tensor_type).astype(self.numpy_dtype)).view(self.d, self.n_head_q * self.d_v)  # [d, n_head_q * d_v]

            attn_norm_tensor = self.get_tensor_by_name(f"blk.{i}.attn_norm.weight")
            self.attn_norm_weight[i] = torch.from_numpy(dequantize(
                attn_norm_tensor.data.copy(), attn_norm_tensor.tensor_type).astype(np.float32)).view(self.d)                            # [d]

        # Load feed-forward network weights
        for i in range(self.n_block):
            ffn_gate_tensor = self.get_tensor_by_name(f"blk.{i}.ffn_gate.weight")
            self.ffn_gate_weight[i] = torch.from_numpy(dequantize(
                ffn_gate_tensor.data.copy(), ffn_gate_tensor.tensor_type).astype(self.numpy_dtype)).view(self.d_ffn, self.d)            # [d_ffn, d]

            ffn_up_tensor = self.get_tensor_by_name(f"blk.{i}.ffn_up.weight")
            self.ffn_up_weight[i] = torch.from_numpy(dequantize(
                ffn_up_tensor.data.copy(), ffn_up_tensor.tensor_type).astype(self.numpy_dtype)).view(self.d_ffn, self.d)                # [d_ffn, d]

            ffn_down_tensor = self.get_tensor_by_name(f"blk.{i}.ffn_down.weight")
            self.ffn_down_weight[i] = torch.from_numpy(dequantize(
                ffn_down_tensor.data.copy(), ffn_down_tensor.tensor_type).astype(self.numpy_dtype)).view(self.d, self.d_ffn)            # [d, d_ffn]

            ffn_norm_tensor = self.get_tensor_by_name(f"blk.{i}.ffn_norm.weight")
            self.ffn_norm_weight[i] = torch.from_numpy(dequantize(
                ffn_norm_tensor.data.copy(), ffn_norm_tensor.tensor_type).astype(np.float32)).view(self.d)                              # [d]

        # Load RoPE frequencies
        rope_freqs_tensor = self.get_tensor_by_name("rope_freqs.weight")
        self.rope_freqs_weight.copy_(torch.from_numpy(dequantize(
            rope_freqs_tensor.data.copy(), rope_freqs_tensor.tensor_type).astype(np.float32)).view(self.d_rope // 2))                   # [d_rope // 2]
        self.rope_theta = (self.rope_freqs_weight * self.freq_base) ** (
            -2 * torch.arange(0, self.d_rope // 2, dtype=torch.float32, device=self.device) / self.d_rope)                              # [d_rope // 2]

        # Load output normalization weight
        output_norm_tensor = self.get_tensor_by_name("output_norm.weight")
        self.output_norm_weight.copy_(torch.from_numpy(dequantize(
            output_norm_tensor.data.copy(), output_norm_tensor.tensor_type).astype(np.float32)).view(self.d))                           # [d]

    def forward(self, inputs: torch.Tensor, use_kvcache: bool = True):
        seq_len = inputs.shape[0]
        attn_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))  # [sl, sl]
        hidden_states = self.token_embed_weight[inputs]                                               # [sl, d]
        for i in range(self.n_block):
            # RMS Normalization
            identity = hidden_states
            hidden_states = self.rmsnorm(hidden_states, self.attn_norm_weight[i])                     # [sl, d]
            # Attention
            # q: [nhq, sl, dk] = RoPE([sl, d] @ [nhq, dk, d])
            q = self.rope(hidden_states @ self.attn_q_weight[i].transpose(1, 2), pos=self.ctx_len)    # [nhq, sl, dk]
            # k: [nhkv, sl, dk] = RoPE([sl, d] @ [nhkv, dk, d])
            k = self.rope(hidden_states @ self.attn_k_weight[i].transpose(1, 2), pos=self.ctx_len)    # [nhkv, sl, dk]
            if use_kvcache:
                self.kcache[i, :, self.ctx_len:self.ctx_len + seq_len] = k
            # v: [nhkv, sl, dv] = [sl, d] @ [nhkv, dv, d]
            v = hidden_states @ self.attn_v_weight[i].transpose(1, 2)                                 # [nhkv, sl, dv]
            if use_kvcache:
                self.vcache[i, :, self.ctx_len:self.ctx_len + seq_len] = v
            if use_kvcache:
                k = self.kcache[i, :, :self.ctx_len + seq_len].repeat_interleave(self.n_head_q // self.n_head_kv, dim=0) # [nhq, sl, dk]
                v = self.vcache[i, :, :self.ctx_len + seq_len].repeat_interleave(self.n_head_q // self.n_head_kv, dim=0) # [nhq, sl, dv]
            else:
                k = k.repeat_interleave(self.n_head_q // self.n_head_kv, dim=0)                       # [nhq, sl, dk]
                v = v.repeat_interleave(self.n_head_q // self.n_head_kv, dim=0)                       # [nhq, sl, dv]
            # Attention scores: [nhq, sl, sl] = [nhq, sl, dk] @ [nhkv, sl, dk]
            scores = torch.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)                              # [nhq, sl, sl]
            scores = scores.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))                       # [nhq, sl, sl]
            scores = torch.softmax(scores, dim=-1)                                                    # [nhq, sl, sl]
            # Attention output: [nhq, sl, dv] = [nhq, sl, sl] @ [nhq, sl, dv]
            o = torch.bmm(scores, v)                                                                  # [nhq, sl, dv]
            # Output projection: [sl, d] += [nhq, sl, dv] @ [d, nhq * dv]
            hidden_states = identity + o.transpose(0, 1).reshape(-1, self.n_head_q * self.d_v) @ self.attn_o_weight[i].T

            # RMS Normalization
            identity = hidden_states
            hidden_states = self.rmsnorm(hidden_states, self.ffn_norm_weight[i])                      # [sl, d]
            # Feed Forward Network
            # Gate: [sl, d_ffn] = [sl, d] @ [dffn, d]
            gate = hidden_states @ self.ffn_gate_weight[i].T                                          # [sl, d_ffn]
            gate = torch.nn.functional.silu(gate)
            # Up: [sl, d_ffn] = [sl, d] @ [dffn, d]
            up = hidden_states @ self.ffn_up_weight[i].T                                              # [sl, d_ffn]
            # Down: [sl, d] = [sl, d_ffn] @ [d, dffn]
            down = (gate * up) @ self.ffn_down_weight[i].T                                            # [sl, d]
            hidden_states = identity + down
        hidden_states = self.rmsnorm(hidden_states, self.output_norm_weight)                          # [sl, d]
        outputs = hidden_states @ self.token_embed_weight.T                                           # [sl, vocab_size]

        self.ctx_len += seq_len

        return outputs

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor):
        x_fp32 = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.rms_epsilon)
        return ((x_fp32 / norm) * weight).to(self.torch_dtype)

    def rope(self, x: torch.Tensor, pos: int = 0):
        x_fp32 = x.to(torch.float32)                                      # [nh, sl, d_rope]
        idx = torch.arange(x.shape[1], device=x.device) + pos             # [sl]
        idx_theta = torch.einsum('s,d->sd', idx, self.rope_theta)         # [sl, d_rope // 2]
        cos = torch.cos(idx_theta)                                        # [sl, d_rope // 2]
        sin = torch.sin(idx_theta)                                        # [sl, d_rope // 2]
        x_left  = x_fp32[:, :, ::2]                                       # [nh, sl, d_rope // 2]
        x_right = x_fp32[:, :, 1::2]                                      # [nh, sl, d_rope // 2]
        y_left  = x_left * cos - x_right * sin                            # [nh, sl, d_rope // 2]
        y_right = x_left * sin + x_right * cos                            # [nh, sl, d_rope // 2]
        return torch.cat((y_left, y_right), dim=-1).to(self.torch_dtype)  # [nh, sl, d_rope]

class Client:
    def __init__(self, model_name: str, gguf_path: str, max_ctx_len: int = 8192, dtype: str = "float16", device: str = "cuda"):
        self.model_name = model_name
        self.gguf_path = gguf_path
        self.max_ctx_len = max_ctx_len
        self.dtype = dtype
        self.device = device

        self.model = Model(gguf_path, max_ctx_len, dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prefill_performance(self, seq_len: int = 1024, ctx_len: int = 0, use_kvcache: bool = True):
        self.model.initialize()

        tokens = torch.zeros((ctx_len + seq_len,), dtype=torch.int64, device=self.device)

        if ctx_len > 0:
            self.model.forward(tokens[:ctx_len], use_kvcache=use_kvcache)

        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        if use_kvcache:
            self.model.forward(tokens[ctx_len:ctx_len + seq_len], use_kvcache=True)
        else:
            self.model.forward(tokens[:ctx_len + seq_len], use_kvcache=False)

        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        print(f"Prefill performance: seq_len = {seq_len}, ctx_len = {ctx_len}, use_kvcache = {use_kvcache}, "
              f"duration = {(end - start) / 1e6:.2f} ms, speed = {seq_len / ((end - start) / 1e9):.2f} tokens/s")

    def decode_performance(self, seq_len: int = 1024, ctx_len: int = 0, use_kvcache: bool = True):
        self.model.initialize()

        tokens = torch.zeros((ctx_len + seq_len,), dtype=torch.int64, device=self.device)

        if ctx_len > 0:
            self.model.forward(tokens[:ctx_len], use_kvcache=use_kvcache)

        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        for i in range(seq_len):
            if use_kvcache:
                outputs = self.model.forward(tokens[ctx_len + i:ctx_len + i + 1], use_kvcache=True)
            else:
                outputs = self.model.forward(tokens[:ctx_len + i + 1], use_kvcache=False)

        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        print(f"Decode performance: seq_len = {seq_len}, ctx_len = {ctx_len}, use_kvcache = {use_kvcache}, "
              f"duration = {(end - start) / 1e6:.2f} ms, speed = {seq_len / ((end - start) / 1e9):.2f} tokens/s")

    def generate(self, question: str, use_kvcache: bool = True):
        self.model.initialize()

        tokens = torch.empty((self.max_ctx_len,), dtype=torch.int64, device=self.device)
        question_tokens = self.tokenizer.encode(question, return_tensors="pt").to(self.device)[0]
        ctx_len = question_tokens.shape[0]
        tokens[:ctx_len] = question_tokens

        torch.cuda.synchronize()
        start = time.perf_counter_ns()

        input_tokens = tokens[:ctx_len]
        while ctx_len < self.max_ctx_len and tokens[ctx_len - 1] != self.tokenizer.eos_token_id:
            outputs = self.model.forward(input_tokens, use_kvcache=use_kvcache)
            next_token_logits = outputs[-1, :]
            next_token_id = next_token_logits.argmax().item()
            tokens[ctx_len] = next_token_id
            ctx_len += 1

            next_token = self.tokenizer.decode(tokens[ctx_len - 1:ctx_len], skip_special_tokens=True)
            print(next_token, end='', flush=True)
            if use_kvcache:
                input_tokens = tokens[ctx_len - 1:ctx_len]
            else:
                input_tokens = tokens[:ctx_len]

        print("\n")

        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        print(f"Generated {ctx_len} tokens in {(end - start) / 1e6:.2f} ms.")
        print(f"Average tokens per second: {ctx_len / ((end - start) / 1e9):.2f} tokens/s.")

if __name__ == "__main__":
    login(token=os.getenv("HF_TOKEN"))

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    gguf_path = "../models/Llama-3.2-1B-Instruct-BF16.gguf"

    client_fp32 = Client(model_name, gguf_path, max_ctx_len=8192, dtype="float32", device="cuda")
    client_fp16 = Client(model_name, gguf_path, max_ctx_len=8192, dtype="float16", device="cuda")

    def print_title(title: str, width: int = 160):
        title_len = len(title) + 2
        front_len = (width - title_len) // 2
        back_len = width - title_len - front_len
        print("=" * front_len, title, "=" * back_len)

    def print_line(width: int = 160):
        print("=" * width)

    question = "What is computer architecture?"
    print_title("Question")
    print(question)
    print_line()

    print_title("Answer FP32")
    print("Generating answer...\n")
    client_fp32.generate(question, use_kvcache=True)
    print_line()

    print_title("Answer FP16")
    print("Generating answer...\n")
    client_fp16.generate(question, use_kvcache=True)
    print_line()

    print_title("Performance FP32")
    client_fp32.prefill_performance(seq_len=1024, ctx_len=0, use_kvcache=True)
    client_fp32.prefill_performance(seq_len=1024, ctx_len=0, use_kvcache=False)
    client_fp32.decode_performance(seq_len=1024, ctx_len=0, use_kvcache=True)
    # client.decode_performance(seq_len=1024, ctx_len=0, use_kvcache=False)
    print_line()

    print_title("Performance FP16")
    client_fp16.prefill_performance(seq_len=1024, ctx_len=0, use_kvcache=True)
    client_fp16.prefill_performance(seq_len=1024, ctx_len=0, use_kvcache=False)
    client_fp16.decode_performance(seq_len=1024, ctx_len=0, use_kvcache=True)
    # client_fp16.decode_performance(seq_len=1024, ctx_len=0, use_kvcache=False)
    print_line()