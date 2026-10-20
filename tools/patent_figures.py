"""Patent drawings for the disclosure (patent/技术交底书_*.md), Chinese practice: black line art only.

Four drawings, no colour, no grey fills, no photographs, the figure number below each drawing:

    图1  系统总框图（定位器、冻结检测器两路推理、选择器、输出选择）
    图2  定位器流程图
    图3  选择器的特征构成与决策流程，含阈值标定
    图4  无填充扰动与逐块补零扰动的对比示意

    python tools/patent_figures.py

Writes patent/figures/figure1..4.{pdf,png} (600 dpi) and patent/附图_2026-09-20.docx.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

FONT = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
LINE = 1.0
OUTPUT = Path("patent/figures")


def canvas(width: float, height: float):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "axes.unicode_minus": False,
                         "pdf.fonttype": 42})
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100 * height / width)
    axis.axis("off")
    return figure, axis


def box(axis, x, y, w, h, text, *, size=9, dashed=False, align="center"):
    axis.add_patch(Rectangle((x, y), w, h, fill=False, lw=LINE, ec="black",
                             linestyle="--" if dashed else "-"))
    axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.5)
    return x + w / 2, y + h / 2


def label(axis, x, y, text, *, size=9, ha="center", va="center"):
    axis.text(x, y, text, ha=ha, va=va, fontsize=size, linespacing=1.5)


def arrow(axis, x1, y1, x2, y2, *, text=None, size=8, offset=(0, 1.6), dashed=False):
    axis.add_patch(FancyArrow(x1, y1, x2 - x1, y2 - y1, width=0.05, head_width=1.4, head_length=1.8,
                              length_includes_head=True, color="black", linewidth=LINE * 0.6,
                              linestyle="--" if dashed else "-"))
    if text:
        axis.text((x1 + x2) / 2 + offset[0], (y1 + y2) / 2 + offset[1], text, ha="center", va="center", fontsize=size)


def elbow(axis, x1, y1, x2, y2, *, via_x=None):
    """Right-angled connector ending with an arrow head."""
    via = x1 if via_x is None else via_x
    axis.plot([x1, via], [y1, y1], color="black", lw=LINE * 0.6)
    axis.plot([via, via], [y1, y2], color="black", lw=LINE * 0.6)
    arrow(axis, via, y2, x2, y2)


def number(figure, text):
    figure.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=10, family="sans-serif")


def figure1() -> plt.Figure:
    figure, axis = canvas(7.0, 8.6)
    top = 100 * 8.6 / 7.0
    label(axis, 26, top - 5, "第一期影像 T1", size=9)
    label(axis, 60, top - 5, "第二期影像 T2", size=9)

    axis.add_patch(Rectangle((10, top - 44), 80, 30, fill=False, lw=LINE, ec="black", linestyle="--"))
    label(axis, 50, top - 11.5, "定位器", size=10)
    box(axis, 15, top - 39, 32, 20, "特征提取\n粗匹配\n两轮细化")
    box(axis, 53, top - 39, 32, 20, "可靠性加权\nHuber-IRLS\n拟合全局相似变换")
    arrow(axis, 47, top - 29, 53, top - 29)
    arrow(axis, 26, top - 8, 26, top - 19)
    arrow(axis, 60, top - 8, 60, top - 19)

    box(axis, 22, top - 62, 56, 9, "用变换参数 M 校正第二期影像，得到 T2′")
    arrow(axis, 50, top - 44, 50, top - 53, text="变换参数 M", size=8, offset=(11, 0))

    box(axis, 4, top - 82, 42, 11, "冻结检测器\n原始输出 D(T1, T2)")
    box(axis, 54, top - 82, 42, 11, "冻结检测器\n校正输出 D(T1, T2′)")
    arrow(axis, 70, top - 62, 70, top - 71)                       # 校正影像 → 右路
    axis.plot([2, 2], [top - 12, top - 76.5], color="black", lw=LINE * 0.6)    # 原始影像对 → 左路
    axis.plot([2, 26], [top - 12, top - 12], color="black", lw=LINE * 0.6)
    arrow(axis, 2, top - 76.5, 4, top - 76.5)

    box(axis, 18, top - 104, 64, 14, "选择器\n配准特征 + 任务特征 → 采用概率 p")
    arrow(axis, 25, top - 82, 35, top - 90)
    arrow(axis, 75, top - 82, 65, top - 90)
    axis.plot([98, 98], [top - 29, top - 97], color="black", lw=LINE * 0.6, linestyle="--")
    axis.plot([85, 98], [top - 29, top - 29], color="black", lw=LINE * 0.6, linestyle="--")
    arrow(axis, 98, top - 97, 82, top - 97, dashed=True)
    label(axis, 92, top - 92, "配准统计量", size=8)

    box(axis, 10, top - 118, 80, 10, "p ≥ τ：输出 D(T1, T2′)      p < τ：输出 D(T1, T2)")
    arrow(axis, 50, top - 104, 50, top - 108)
    label(axis, 77, top - 106.5, "阈值 τ 在留出的已配准影像上标定", size=8, ha="center")
    number(figure, "图1")
    return figure


def figure2() -> plt.Figure:
    figure, axis = canvas(6.2, 6.6)
    top = 100 * 6.6 / 6.2
    steps = [("冻结骨干网络提取两期影像的多尺度特征", 0),
             ("低分辨率特征上互近邻粗匹配，得到对应点与置信度 c", 1),
             ("按当前变换对齐特征，局部窗口内重搜索对应点（两轮）", 2),
             ("可靠性分支预测变化分数 s，权重 w = c × sigmoid(-s)", 3),
             ("以 w 加权、带 Huber 损失的迭代重加权最小二乘\n拟合全局相似变换，输出变换参数 M", 4)]
    height, gap = 9.5, 4.0
    for text, index in steps:
        y = top - 10 - index * (height + gap)
        box(axis, 8, y - height, 84, height, text, size=9)
        if index:
            arrow(axis, 50, y + gap, 50, y)
    y_last = top - 10 - 4 * (height + gap) - height
    box(axis, 8, y_last - 13, 84, 9.5, "交换两期影像角色再估计一次，得到反向变换 M′\n以 M 与 M′ 的复合偏差作为前后向一致性指标", size=9)
    arrow(axis, 50, y_last, 50, y_last - 3.5)
    number(figure, "图2")
    return figure


def figure3() -> plt.Figure:
    figure, axis = canvas(6.6, 6.2)
    top = 100 * 6.2 / 6.6
    box(axis, 4, top - 30, 28, 22, "配准特征（6 维）\n错位概率、估计位移\n对数尺度、中心平移\n前后向一致性\n有效像素占比", size=8.5)
    box(axis, 4, top - 62, 28, 26, "任务特征（13 维）\n两路检测输出的\n平均概率、变化占比\n熵、置信度\n概率差异、二值差异\n交并比", size=8.5)
    box(axis, 42, top - 46, 24, 16, "小型多层感知机\n32-16，约 1249 参数", size=9)
    arrow(axis, 32, top - 19, 42, top - 34)
    arrow(axis, 32, top - 49, 42, top - 42)
    arrow(axis, 66, top - 38, 74, top - 38)
    box(axis, 74, top - 54, 24, 28, "采用概率 p\n与阈值 τ 比较\n\np ≥ τ：采用校正结果\np < τ：保留原始结果", size=8.5)

    box(axis, 4, top - 86, 48, 16, "阈值标定：在留出的已配准影像上遍历阈值网格，\n取使已配准影像精度下降不超过\n给定上限的最小阈值", size=8.5)
    axis.plot([52, 86], [top - 78, top - 78], color="black", lw=LINE * 0.6, linestyle="--")
    axis.plot([86, 86], [top - 78, top - 58], color="black", lw=LINE * 0.6, linestyle="--")
    arrow(axis, 86, top - 58, 86, top - 54, dashed=True)
    number(figure, "图3")
    return figure


def tile(axis, x, y, size, *, shift=0.0, scale=1.0, dashed=False):
    axis.add_patch(Rectangle((x, y), size, size, fill=False, lw=LINE, ec="black",
                             linestyle="--" if dashed else "-"))


def figure4() -> plt.Figure:
    figure, axis = canvas(7.0, 3.8)
    top = 100 * 3.8 / 7.0

    label(axis, 25, top - 4, "（a）现有做法：在小块内变换并补零", size=9)
    axis.add_patch(Rectangle((5, top - 32), 16, 16, fill=False, lw=LINE, ec="black"))
    label(axis, 13, top - 35.5, "原始小块", size=8)
    arrow(axis, 23, top - 24, 30, top - 24)
    label(axis, 26.5, top - 19.5, "小块内变换\n空缺处补零", size=7.5)
    axis.add_patch(Rectangle((32, top - 32), 16, 16, fill=False, lw=LINE, ec="black"))
    axis.add_patch(Rectangle((32, top - 32), 16, 16, fill=False, lw=0.5, ec="black", hatch="///"))
    axis.add_patch(Rectangle((35, top - 29), 13, 13, fill=True, fc="white", lw=LINE, ec="black"))
    label(axis, 40, top - 35.5, "阴影为补零区域，其形状\n编码了平移量与缩放量", size=7.5)

    label(axis, 75, top - 4, "（b）本方案：先整幅变换再裁切", size=9)
    axis.add_patch(Rectangle((54, top - 34), 24, 24, fill=False, lw=LINE, ec="black"))
    label(axis, 66, top - 37.5, "整幅影像变换后", size=8)
    axis.add_patch(Rectangle((59, top - 28), 12, 12, fill=False, lw=LINE, ec="black", linestyle="--"))
    label(axis, 65, top - 22, "裁切窗口", size=7.5)
    arrow(axis, 80, top - 22, 86, top - 22)
    axis.add_patch(Rectangle((88, top - 28), 12, 12, fill=False, lw=LINE, ec="black"))
    label(axis, 94, top - 31.5, "小块内无填充", size=7.5)
    number(figure, "图4")
    return figure


def word(paths, output: Path, captions) -> Path:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    document = Document()
    for index, (path, caption) in enumerate(zip(paths, captions)):
        picture = document.add_paragraph()
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.add_run().add_picture(str(path), width=Cm(15.0))
        text = document.add_paragraph()
        text.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = text.add_run(caption)
        run.font.size = Pt(10.5)
        run.font.name = "SimSun"
        if index != len(paths) - 1:
            document.add_page_break()
    document.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--word", default="patent/附图_2026-09-20.docx")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    builders = (figure1, figure2, figure3, figure4)
    captions = ("图1  系统总框图", "图2  定位器流程图", "图3  选择器的特征构成、决策与阈值标定",
                "图4  逐块补零扰动与无填充扰动的对比")
    pngs = []
    for index, builder in enumerate(builders, start=1):
        figure = builder()
        for suffix, options in ((".pdf", {}), (".png", {"dpi": 600})):
            path = output_dir / f"figure{index}{suffix}"
            figure.savefig(path, facecolor="white", **options)
            if suffix == ".png":
                pngs.append(path)
        plt.close(figure)
        print(output_dir / f"figure{index}.png")
    print(word(pngs, Path(args.word), captions))


if __name__ == "__main__":
    main()
