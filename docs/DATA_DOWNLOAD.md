# 数据下载与放置

## 第一阶段只需要 LEVIR-CD

官方主页：<https://justchenhao.github.io/LEVIR/>

请从官方页面或其明确指向的官方存储下载，保留压缩包、README、许可说明和下载日期。不要从不明镜像获得数据后直接当作官方版本，也不要手工修改标签。

下载后推荐保持如下结构：

```text
data/raw/LEVIR-CD/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

部分版本使用：

```text
LEVIR-CD/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

两种结构均受 `tools/build_manifest.py` 支持。若下载包中的目录名不同，不要直接批量改名；先记录实际目录树，再通过命令行参数指定。

## 原始数据保护

- `data/raw` 不进入 Git。
- 不在原始目录里生成裁剪图、预测图或缓存。
- 不覆盖原标签。
- 数据划分以官方文件为准。
- 在运行训练前先执行 manifest 构建和数据审计。

