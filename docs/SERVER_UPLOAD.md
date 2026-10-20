# 服务器上传与首次运行

## 1. 上传内容

推荐在服务器建立：

```text
remote_project/
├── rs_cd_project/
│   ├── configs/
│   ├── docs/
│   ├── manifests/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── tools/
│   ├── pyproject.toml
│   └── README.md
├── data/
│   └── LEVIR-CD/
│       ├── train/{A,B,label}/
│       ├── val/{A,B,label}/
│       └── test/{A,B,label}/
└── pretrained/
    ├── BIT_LEVIR/best_ckpt.pt
    └── PAM-pretrain/
        ├── pam_net_A.pth
        └── pam_net_F.pth
```

不需要上传：

- `rs_cd_project/data/synthetic/`；
- `rs_cd_project/outputs/` 中的本机 CPU smoke；
- `.pytest_cache`、`__pycache__`；
- 老师原始三套源码；
- PDF 渲染临时文件。

建议保留并上传：

- `manifests/datasets/levir_cd.csv`；
- `reports/data_audit_LEVIR-CD.json`；
- `reports/data_audit_LEVIR-CD.md`；
- 三张 LEVIR-CD 抽查拼图；
- 原始数据来源和许可说明。

manifest 中是相对路径，所以只要配置的 `dataset_root` 指向服务器上的
`LEVIR-CD` 根目录，就可以复用。

## 2. 三份权重校验值

```text
best_ckpt.pt
C159BA76143447F58C9F367CE8126A0014F2E4BA218CDB97CCA173952C38CB3B

pam_net_A.pth
C20BBBF2B2448FC08F8A87DA5384A246AC0C48B20A63F5A03C8038DC4A2E85D0

pam_net_F.pth
78ABC9DD6BE7C577F98B892CF6353D4F48EF36211E3C009A681FF7143D3EBB4B
```

服务器可用：

```bash
sha256sum pretrained/BIT_LEVIR/best_ckpt.pt
sha256sum pretrained/PAM-pretrain/pam_net_A.pth
sha256sum pretrained/PAM-pretrain/pam_net_F.pth
```

`best_ckpt.pt` 不能被当前 PyTorch 的 `weights_only=True` 直接读取，因为其中包含
NumPy 标量元数据。除非已确认下载来源可信且使用对应官方代码，否则不要用
`weights_only=False` 随意加载。两份 PAM 权重可以在安全模式下读取为 state dict。

## 3. 路径配置

训练和测试命令使用 `--dataset-root ../data/LEVIR-CD` 覆盖本地配置路径。覆盖后的真实路径会写入每次运行的 `config_resolved.yaml`，不需要直接编辑基础配置，也不要修改 manifest 内部的相对路径。

## 4. 环境与单元测试

```bash
cd remote_project/rs_cd_project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 先按服务器CUDA版本安装PyTorch，再执行：
pip install -e ".[dev]"

python -m tools.capture_environment \
  --output environment/system_info_server.json

pytest -q
```

## 5. 48GB GPU 首次探测

先不要直接用最大 batch。依次测试：

```bash
python -m tools.gpu_smoke \
  --config configs/p1_fc_siam_diff.yaml \
  --batch-size 8 \
  --image-size 256 \
  --steps 10 \
  --output reports/gpu_smoke_bs8.json

python -m tools.gpu_smoke \
  --config configs/p1_fc_siam_diff.yaml \
  --batch-size 16 \
  --image-size 256 \
  --steps 10 \
  --output reports/gpu_smoke_bs16.json

python -m tools.gpu_smoke \
  --config configs/p1_fc_siam_diff.yaml \
  --batch-size 32 \
  --image-size 256 \
  --steps 10 \
  --output reports/gpu_smoke_bs32.json
```

选择峰值 reserved memory 不超过可用显存约 80% 的 batch，给验证、日志和环境波动留余量。首个正式复现实验仍建议保持配置中的 batch size 8，除非同时按线性规则调整学习率并把它登记为新实验协议。

### RAM 缓存

基础配置启用：

```yaml
data:
  num_workers: 8
  prefetch_factor: 4
  cache_mode: ram
```

它会在 DataLoader worker 启动前，将对应 split 的完整 RGB/RGB/mask 解码为
`uint8` 数组并缓存到系统内存。训练集约占 3.0 GiB，验证集约占 0.4 GiB；
随机裁剪、翻转和旋转仍在每次取样时执行，不会固定为一份预处理 crop。

RAM 缓存建议只在 Linux 服务器使用。Linux 默认 fork worker，可通过
copy-on-write 共享只读缓存；Windows spawn worker 可能复制缓存。若系统内存不足，
将 `cache_mode` 改回 `none`。

### 关于 batch 128

batch 32 实测峰值约 9.5 GiB 时，单卡 batch 128 预计超过标准 RTX 4090 的
24 GiB。两张 4090 的总显存不会在当前单卡代码中自动合并。即使使用双卡 DDP，
global batch 128 也会让 445 张训练父影像每轮只有约 4 次参数更新，必须重新设计
学习率、epoch 和统一比较协议。因此正式配置应先选择 batch 32，batch 64 仅作为
容量/吞吐探测，不直接替换论文协议。

## 6. 真实数据验收顺序

```bash
python -m tools.audit_dataset \
  --dataset-root ../data/LEVIR-CD \
  --manifest manifests/datasets/levir_cd.csv \
  --mask-threshold 127 \
  --output-json reports/data_audit_LEVIR-CD-server.json \
  --output-md reports/data_audit_LEVIR-CD-server.md

python -m tools.train \
  --config configs/p1_fc_siam_diff.yaml \
  --dataset-root ../data/LEVIR-CD \
  --device cuda \
  --overfit-samples 20
```

20 样本过拟合通过后，才运行完整训练：

```bash
python -m tools.train \
  --config configs/p1_fc_siam_diff.yaml \
  --dataset-root ../data/LEVIR-CD \
  --device cuda
```
