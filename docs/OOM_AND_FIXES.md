# 显存不足（OOM）报错与处理说明

本文汇总 VLA 蒸馏训练中遇到的显存相关报错及对应处理方式。

---

## 一、两类 OOM 报错

### 1. 教师 VAE 编码时 OOM（图像帧数过多）

**现象**：在 `teacher_extractor` 内部，Dream Zero 的 **VAE 编码视频** 时崩掉，报错类似：

```text
File ".../wan_video_vae.py", line 582, in forward
    x = F.pad(x, padding)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 500.00 MiB. GPU 0 has ... 364.62 MiB is free.
```

**原因**：在「只传图像+文本」模式（`--teacher_image_text_only`）下，为了满足 Dream Zero 对 `num_image_blocks >= 1` 的要求，我们把 2 帧扩成了 **33 帧**。教师 VAE 要对这 33 帧做编码，显存占用很大，单卡容易 OOM。

**处理**：  
不再按 33 帧扩，只扩到**最少需要的 9 帧**（满足 `(9+3)%4==0` 且 `T_latent=3` → `num_image_blocks=1`），在保证教师前向合法的前提下明显省显存。

- **代码位置**：`cosmos_policy/pipeline/moe_vla_pipeline.py`  
  - 常量 `TEACHER_MIN_FRAMES = 9`  
  - `image_text_only` 分支里用 `target_T = max(T, TEACHER_MIN_FRAMES)`，最多扩到 9 帧；  
  - 非 image_text_only 时，2 帧也改为扩成 9 帧（不再扩成 5 帧，避免 num_image_blocks=0 的 reshape 报错）。

---

### 2. Optimizer.step 时 OOM（显存碎片）

**现象**：前向、反向都跑完，在 **更新参数** 时崩掉，报错类似：

```text
scaler.step(optimizer)
  → optimizer.step()
    → adam() in torch/optim/adam.py
      → _multi_tensor_adam()
        → exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)   ← 这里 OOM
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has ... 8.00 MiB is free.
```

**具体位置**：**Adam 的 step 内部**，在算「二阶矩的平方根」`sqrt(exp_avg_sq)` 时，需要一块约 16～20 MiB 的临时显存；此时整卡已被模型参数、梯度、优化器状态（exp_avg、exp_avg_sq）占满，只剩 8 MiB 且可能碎片化，连续 16 MiB 分配失败。

**原因**：  
- 总显存（例如 80GiB）几乎被模型参数、梯度、优化器状态（Adam 的 exp_avg、exp_avg_sq）以及前向/反向的中间结果占满。  
- backward 结束后，中间激活会释放，但参数、梯度、优化器状态仍在。  
- 若显存被多次分配/释放后产生**碎片**，PyTorch 可能报告“只剩 12 MiB 空闲”，但实际需要的那 20 MiB 无法找到一块连续空间，从而在 `optimizer.step()` 里做 `sqrt(exp_avg_sq)` 时 OOM。

**处理**：  

1. **在 step 前同步 + 释放缓存、减轻碎片**  
   在每次真正执行 `scaler.step(optimizer)` 之前先 `torch.cuda.synchronize()`（等所有 CUDA 操作结束、释放临时缓冲），再 `torch.cuda.empty_cache()`，让已释放的块尽量还给 CUDA，便于 step 里 `sqrt(exp_avg_sq)` 的 16 MiB 分配成功。

   - **代码位置**：`train_vla_distill.py`，在 `scaler.step(optimizer)` 前：
     ```python
     if torch.cuda.is_available():
         torch.cuda.synchronize()
         torch.cuda.empty_cache()
     scaler.step(optimizer)
     ```

2. **建议设置环境变量，减少碎片**  
   运行前设置：
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
   这样分配器更倾向于使用可扩展的显存段，有利于降低碎片、提高 step 时分配成功率。

3. **仍 OOM 时的进一步手段**  
   - 加大梯度累积步数（如 `--grad_accumulation_steps 32`），等效 batch 不变，单次 backward 的峰值略降。  
   - 使用 `--teacher_image_text_only`，教师只跑图像+文本分支，省教师侧显存。  
   - 确保单卡训练时只占一张卡，避免多卡各 80G 仍把单卡打满。

---

## 二、小结表

| 报错发生位置           | 直接原因           | 处理方式 |
|------------------------|--------------------|----------|
| 教师 VAE 编码（encode） | 图像被扩成 33 帧，编码显存过大 | 只扩到 9 帧（`TEACHER_MIN_FRAMES=9`） |
| optimizer.step（Adam） | 显存几乎用满 + 碎片，step 时 20 MiB 分配失败 | step 前 `empty_cache()` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，必要时加大 grad_accumulation 或 teacher_image_text_only |

以上两处修改都在当前仓库中；若仍 OOM，可再结合 `nvidia-smi`、梯度累积和 `--teacher_image_text_only` 进一步降显存占用。
