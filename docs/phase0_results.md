===== TEACHER MODEL: /home/original_models/Qwen2.5-Coder-3B =====
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████| 434/434 [00:04<00:00, 104.24it/s, Materializing param=model.norm.weight]

Model hidden size: 2048
Number of layers: 36

Found 5 function bodies to mask:
  [function_body] chars 173-286: self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
       ...
  [function_body] chars 333-626: if key not in self._cache:
            return None
        entry = self._cache[k...
  [function_body] chars 676-923: if len(self._cache) >= self.max_size:
            evicted = self._access_order.p...
  [function_body] chars 972-1040: return (__import__("time").time() - entry["ts"]) > self.ttl...
  [function_body] chars 1083-1136: return hashlib.md5(key.encode()).hexdigest()...

--- Per-token hidden states (last layer) ---

TARGETS
  function_body: shape=torch.Size([28, 2048]), tokens=28
  function_body: shape=torch.Size([62, 2048]), tokens=62
  function_body: shape=torch.Size([63, 2048]), tokens=63
  function_body: shape=torch.Size([19, 2048]), tokens=19
  function_body: shape=torch.Size([10, 2048]), tokens=10

--- Mean-pooled hidden states (last layer) ---
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])

Cosine sim (func 0 vs func 1): 0.9844
(If ~1.0 for all pairs, try middle layers instead of last)

--- Layer comparison (cosine sim between func 0 and func 1) ---
  Layer   1: cosine_sim = 0.9370
  Layer   9: cosine_sim = 0.8955
  Layer  18: cosine_sim = 0.8638
  Layer  27: cosine_sim = 0.8833
  Layer  36: cosine_sim = 0.9844
===== TEACHER MODEL: /home/original_models/Qwen2.5-Coder-3B-Instruct =====
Loading weights: 100%|█████████████████████████████████████████████████████████| 434/434 [00:04<00:00, 99.72it/s, Materializing param=model.norm.weight]

Model hidden size: 2048
Number of layers: 36

Found 5 function bodies to mask:
  [function_body] chars 173-286: self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
       ...
  [function_body] chars 333-626: if key not in self._cache:
            return None
        entry = self._cache[k...
  [function_body] chars 676-923: if len(self._cache) >= self.max_size:
            evicted = self._access_order.p...
  [function_body] chars 972-1040: return (__import__("time").time() - entry["ts"]) > self.ttl...
  [function_body] chars 1083-1136: return hashlib.md5(key.encode()).hexdigest()...

--- Per-token hidden states (last layer) ---

TARGETS
  function_body: shape=torch.Size([28, 2048]), tokens=28
  function_body: shape=torch.Size([62, 2048]), tokens=62
  function_body: shape=torch.Size([63, 2048]), tokens=63
  function_body: shape=torch.Size([19, 2048]), tokens=19
  function_body: shape=torch.Size([10, 2048]), tokens=10

--- Mean-pooled hidden states (last layer) ---
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])
  function_body: shape=torch.Size([1, 2048])

Cosine sim (func 0 vs func 1): 0.8564
(If ~1.0 for all pairs, try middle layers instead of last)

--- Layer comparison (cosine sim between func 0 and func 1) ---
  Layer   1: cosine_sim = 0.9229
  Layer   9: cosine_sim = 0.8770
  Layer  18: cosine_sim = 0.8633
  Layer  27: cosine_sim = 0.8579
  Layer  36: cosine_sim = 0.8564
===== TEACHER MODEL: /home/original_models/Qwen3-8B =====
Loading weights: 100%|███████████████████████████| 399/399 [00:09<00:00, 40.44it/s, Materializing param=model.norm.weight]

Model hidden size: 4096
Number of layers: 36

Found 5 function bodies to mask:
  [function_body] chars 173-286: self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
       ...
  [function_body] chars 333-626: if key not in self._cache:
            return None
        entry = self._cache[k...
  [function_body] chars 676-923: if len(self._cache) >= self.max_size:
            evicted = self._access_order.p...
  [function_body] chars 972-1040: return (__import__("time").time() - entry["ts"]) > self.ttl...
  [function_body] chars 1083-1136: return hashlib.md5(key.encode()).hexdigest()...

--- Per-token hidden states (last layer) ---

TARGETS
  function_body: shape=torch.Size([28, 4096]), tokens=28
  function_body: shape=torch.Size([62, 4096]), tokens=62
  function_body: shape=torch.Size([63, 4096]), tokens=63
  function_body: shape=torch.Size([19, 4096]), tokens=19
  function_body: shape=torch.Size([10, 4096]), tokens=10

--- Mean-pooled hidden states (last layer) ---
  function_body: shape=torch.Size([1, 4096])
  function_body: shape=torch.Size([1, 4096])
  function_body: shape=torch.Size([1, 4096])
  function_body: shape=torch.Size([1, 4096])
  function_body: shape=torch.Size([1, 4096])

Cosine sim (func 0 vs func 1): 0.9058
(If ~1.0 for all pairs, try middle layers instead of last)

--- Layer comparison (cosine sim between func 0 and func 1) ---
  Layer   1: cosine_sim = 0.9834
  Layer   9: cosine_sim = 0.9438
  Layer  18: cosine_sim = 0.8765
  Layer  27: cosine_sim = 0.8799
  Layer  36: cosine_sim = 0.9058