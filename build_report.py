"""Build the course report directly from saved experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
REPORT_PATH = REPORTS / "GTSRB改进ResNet18大作业报告_仓库地址已补.docx"
VARIANTS = ["baseline", "cbam", "multiscale", "gelu", "mish", "groupnorm", "combined"]
DISPLAY_NAMES = {
    "baseline": "ResNet 18基线",
    "cbam": "CBAM",
    "multiscale": "多尺度卷积",
    "gelu": "GELU",
    "mish": "Mish",
    "groupnorm": "GroupNorm",
    "combined": "多尺度卷积加Mish",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


metrics = {name: load_json(OUTPUTS / name / "metrics.json") for name in VARIANTS}
config = load_json(OUTPUTS / "baseline" / "config.json")
robustness = {
    name: load_json(OUTPUTS / name / "robustness.json")
    for name in ("baseline", "combined")
}


def set_run_font(run, latin="Times New Roman", east_asia="宋体", size=10.5, bold=None):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.bold = bold


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=90, bottom=90, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_keep_with_next(paragraph, value=True):
    paragraph.paragraph_format.keep_with_next = value


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_run_font(run, size=9)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, end])
    tail = paragraph.add_run(" 页")
    set_run_font(tail, size=9)


def add_body(doc, text, first_indent=True, bold_lead=None):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    paragraph.paragraph_format.space_after = Pt(4)
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(21)
    if bold_lead and text.startswith(bold_lead):
        first = paragraph.add_run(bold_lead)
        set_run_font(first, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)
    return paragraph


def add_bullet(doc, text):
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.left_indent = Pt(21)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.line_spacing = 1.25
    set_run_font(paragraph.add_run(text))
    return paragraph


def add_code(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Pt(18)
    paragraph.paragraph_format.space_after = Pt(1.5)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    set_run_font(run, latin="Consolas", east_asia="等线", size=8.5)
    return paragraph


def add_heading(doc, text, level=1):
    paragraph = doc.add_heading(text, level=level)
    set_keep_with_next(paragraph)
    return paragraph


def add_caption(doc, text, keep_with_next=False):
    paragraph = doc.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = keep_with_next
    run = paragraph.add_run(text)
    set_run_font(run, size=9)
    return paragraph


def add_figure(doc, image_path: Path, caption: str, width=6.55):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    add_caption(doc, caption)


def add_table(doc, headers, rows, caption, widths=None, font_size=8.5):
    add_caption(doc, caption, keep_with_next=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for index, (cell, text) in enumerate(zip(header.cells, headers)):
        set_cell_shading(cell, "1F4E78")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(str(text))
        set_run_font(run, east_asia="微软雅黑", size=font_size, bold=True)
        run.font.color.rgb = RGBColor(255, 255, 255)
        if widths:
            cell.width = Inches(widths[index])
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for index, (cell, value) in enumerate(zip(cells, values)):
            if row_index % 2:
                set_cell_shading(cell, "EAF2F8")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(str(value))
            set_run_font(run, size=font_size)
            if widths:
                cell.width = Inches(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(0.72)
section.bottom_margin = Inches(0.68)
section.left_margin = Inches(0.78)
section.right_margin = Inches(0.78)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(10.5)
for style_name, size, east_asia in (
    ("Title", 22, "微软雅黑"),
    ("Heading 1", 15, "黑体"),
    ("Heading 2", 12, "黑体"),
    ("Heading 3", 11, "黑体"),
):
    style = styles[style_name]
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.bold = style_name != "Title"
styles["Heading 1"].paragraph_format.space_before = Pt(12)
styles["Heading 1"].paragraph_format.space_after = Pt(6)
styles["Heading 2"].paragraph_format.space_before = Pt(8)
styles["Heading 2"].paragraph_format.space_after = Pt(4)

footer = section.footer
add_page_number(footer.paragraphs[0])

# Cover page
doc.add_paragraph().paragraph_format.space_after = Pt(65)
title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(18)
run = title.add_run("基于多尺度卷积与注意力机制的ResNet 18交通标志识别实验报告")
set_run_font(run, east_asia="微软雅黑", size=22, bold=True)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(70)
run = subtitle.add_run("深度学习课程大作业")
set_run_font(run, east_asia="微软雅黑", size=15)

for label in ("学生姓名", "学号", "课程名称", "任课教师"):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(12)
    run = paragraph.add_run(f"{label}  ____________________")
    set_run_font(run, east_asia="宋体", size=12)

date_paragraph = doc.add_paragraph()
date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_paragraph.paragraph_format.space_before = Pt(35)
set_run_font(date_paragraph.add_run("2026年11月5日"), size=11)
doc.add_page_break()

# Abstract
add_heading(doc, "摘要", 1)
add_body(
    doc,
    "本项目在GTSRB德国交通标志数据集上实现适配32乘32图像的ResNet 18，并围绕卷积感受野、注意力机制、激活函数和归一化层开展七组对照实验。所有实验使用相同的固定数据划分、随机种子、增强策略和优化器配置，测试集只在验证集选定最佳权重后使用。标准ResNet 18在12630张测试图像上取得97.9572%的准确率和96.2381%的Macro F1，是七个模型中测试性能和鲁棒性最好的方案。多尺度卷积加Mish组合模型取得最高验证准确率98.8435%，但测试准确率为97.3793%，说明验证集改进没有转化为更好的跨域泛化。CBAM和GroupNorm在当前大批量训练设置下均降低性能。实验结果表明，结构复杂度和验证集峰值不能替代独立测试集、类别均衡指标、实际延迟与扰动测试。",
)
keywords = doc.add_paragraph()
keywords.paragraph_format.space_before = Pt(6)
lead = keywords.add_run("关键词 ")
set_run_font(lead, bold=True)
set_run_font(keywords.add_run("交通标志识别  ResNet 18  多尺度卷积  CBAM  Mish  模型鲁棒性"))

add_heading(doc, "报告结构", 1)
for item in (
    "1 研究目标与主要结论",
    "2 数据集与实验设计",
    "3 基线网络与结构修改",
    "4 实验结果与分析",
    "5 场景应用与可解释性",
    "6 复现方法与代码仓库",
    "7 局限与后续工作",
    "参考文献",
):
    add_bullet(doc, item)
doc.add_page_break()

add_heading(doc, "1 研究目标与主要结论", 1)
add_body(
    doc,
    "本项目希望回答三个问题。第一，经典残差网络能否在不使用预训练权重的情况下达到GTSRB基准识别水平。第二，多尺度卷积、CBAM、GELU、Mish和GroupNorm分别怎样影响精度、参数量、计算量和速度。第三，验证集上的结构收益能否延伸到独立测试集以及模糊、暗光和噪声场景。",
)
add_body(
    doc,
    "主要结论是保留标准ResNet 18作为部署模型。它的测试准确率为97.9572%，Macro F1为96.2381%，单张推理延迟为3.517毫秒。组合模型虽然将验证准确率提高到98.8435%，但测试准确率下降0.5780个百分点，三种扰动下的平均准确率也从92.7263%下降到91.8475%。这一差异说明模型选择必须同时考虑验证集、测试集和目标场景。",
)

add_heading(doc, "2 数据集与实验设计", 1)
add_heading(doc, "2.1 数据集与划分", 2)
add_body(
    doc,
    "实验采用German Traffic Sign Recognition Benchmark。数据包含43个交通标志类别，使用Udacity公开镜像提供的固定划分：训练集34799张、验证集4410张、测试集12630张。固定划分避免各模型因重新抽样而获得不同难度的数据，同时便于复现。训练图像使用随机裁剪、正负10度旋转以及亮度、对比度和饱和度扰动；交通标志具有方向语义，因此没有使用水平翻转。验证与测试阶段只进行尺寸统一和归一化。",
)
add_table(
    doc,
    ["数据子集", "图像数量", "用途"],
    [
        ["训练集", "34799", "更新模型参数"],
        ["验证集", "4410", "选择最佳轮次和组合模块"],
        ["测试集", "12630", "最终一次独立评估"],
    ],
    caption="表 1 GTSRB固定数据集划分及用途",
    widths=[1.4, 1.3, 3.7],
    font_size=9,
)

add_heading(doc, "2.2 训练配置", 2)
add_body(
    doc,
    "七组实验均从随机初始化开始训练，使用AdamW优化器、余弦退火学习率、交叉熵损失和0.1标签平滑。最大训练轮数为30，若验证准确率连续8轮不提升则提前停止。最佳权重由验证准确率确定，所有复杂度和速度统计均由同一台计算机重新测量。",
)
add_table(
    doc,
    ["项目", "设置", "项目", "设置"],
    [
        ["输入尺寸", "32乘32", "Batch size", "128"],
        ["初始学习率", "0.001", "权重衰减", "0.0001"],
        ["最大轮数", "30", "随机种子", "2026"],
        ["优化器", "AdamW", "学习率调度", "Cosine Annealing"],
        ["硬件", "RTX 4060 Laptop 8GB", "混合精度", "CUDA AMP"],
        ["软件", "PyTorch 2.5.1 CUDA 12.4", "Python", "3.10.22"],
    ],
    caption="表 2 模型训练环境与超参数配置",
    widths=[1.25, 2.05, 1.25, 2.05],
    font_size=8.5,
)

add_heading(doc, "2.3 评价指标", 2)
add_body(
    doc,
    "主要指标包括准确率、Macro Precision、Macro Recall和Macro F1。Macro F1先计算每一类的F1再取平均，因此不会让样本较多的类别掩盖小类别错误。模型复杂度使用可训练参数量和卷积及全连接层MACs表示；速度使用CUDA同步后的batch 1平均推理时间表示。鲁棒性测试使用固定强度的高斯模糊、0.45倍亮度和标准差0.08的高斯噪声。",
)

add_heading(doc, "3 基线网络与结构修改", 1)
add_heading(doc, "3.1 小图像ResNet-18", 2)
add_body(
    doc,
    "标准ImageNet版ResNet 18以7乘7步长2卷积和最大池化开始。对于32乘32交通标志，这种下采样会过早丢失边缘和数字信息。本实现将输入端改为3乘3步长1卷积并移除最大池化，保留四个残差阶段和全局平均池化。基线模型共有11190891个参数，单张32乘32图像计算量为555.44M MACs。",
)
add_figure(doc, OUTPUTS / "baseline" / "training_curves.png", "图 1 ResNet-18基线训练与验证曲线", width=6.45)

add_heading(doc, "3.2 CBAM注意力", 2)
add_body(
    doc,
    "CBAM依次计算通道注意力和空间注意力，并放置在每个残差块第二个归一化层之后。该修改将参数量增加88848，MACs增加0.44M。它能够重新加权特征，但在低分辨率图像上，7乘7空间注意力核和额外串行算子也会增加延迟。",
)

add_heading(doc, "3.3 多尺度卷积", 2)
add_body(
    doc,
    "多尺度模块只替换第三和第四阶段残差块中的第二个3乘3卷积。模块包含膨胀率1和2的两条并行分支，每条输出一半通道，随后沿通道维拼接。两条分支的参数和理论MACs之和与原卷积相同，但感受野不同。该设计用于检验交通标志的局部文字、边缘和整体形状是否需要多尺度特征。",
)

add_heading(doc, "3.4 激活函数与归一化", 2)
add_body(
    doc,
    "激活函数实验只将ReLU替换为GELU或Mish。归一化实验只将BatchNorm替换为最多32组的GroupNorm。每次实验只改变一个因素。依据验证集结果，多尺度卷积和Mish分别是结构组与激活组的最佳方案，因此组合模型采用多尺度卷积、Mish和BatchNorm，没有加入验证表现较差的CBAM。",
)

doc.add_page_break()
add_heading(doc, "4 实验结果与分析", 1)
add_heading(doc, "4.1 准确率与类别均衡性能", 2)
performance_rows = []
for name in VARIANTS:
    row = metrics[name]
    performance_rows.append(
        [
            DISPLAY_NAMES[name],
            str(row["best_epoch"]),
            f"{100 * row['best_val_accuracy']:.3f}",
            f"{100 * row['test_accuracy']:.3f}",
            f"{100 * row['macro_precision']:.3f}",
            f"{100 * row['macro_recall']:.3f}",
            f"{100 * row['macro_f1']:.3f}",
        ]
    )
add_table(
    doc,
    ["模型", "最佳轮次", "验证准确率", "测试准确率", "Macro P", "Macro R", "Macro F1"],
    performance_rows,
    caption="表 3 七组模型的准确率与类别均衡指标（单位：%）",
    widths=[1.42, 0.72, 0.94, 0.94, 0.78, 0.78, 0.82],
    font_size=7.7,
)
add_body(
    doc,
    "基线在测试准确率和Macro F1上均排名第一。组合模型验证准确率最高，但测试准确率仅为97.379%，说明固定验证集与测试集之间仍存在分布差异。Mish的验证峰值为98.798%，测试准确率为97.577%；GELU测试准确率为97.197%。激活函数更平滑并没有保证更好的泛化。CBAM和GroupNorm的Macro F1分别只有93.732%和93.712%，说明它们对少数类别的影响大于总体准确率所显示的差异。",
)
add_figure(doc, OUTPUTS / "figures" / "test_performance.png", "图 2 七组模型的测试准确率与Macro F1", width=6.55)

add_heading(doc, "4.2 参数量计算量与速度", 2)
efficiency_rows = []
for name in VARIANTS:
    row = metrics[name]
    efficiency_rows.append(
        [
            DISPLAY_NAMES[name],
            f"{row['parameters'] / 1e6:.3f}",
            f"{row['macs'] / 1e6:.2f}",
            f"{row['model_size_mb']:.2f}",
            f"{row['latency_ms_batch1']:.3f}",
            f"{row['training_seconds'] / 60:.2f}",
        ]
    )
add_table(
    doc,
    ["模型", "参数量 M", "MACs M", "权重 MB", "延迟 ms", "训练 min"],
    efficiency_rows,
    caption="表 4 七组模型的复杂度与实测效率",
    widths=[1.65, 0.9, 0.9, 0.9, 0.9, 0.9],
    font_size=8,
)
add_body(
    doc,
    "多尺度模型的参数量和MACs与基线相同，但推理延迟从3.517毫秒增加到4.615毫秒，组合模型进一步增加到5.272毫秒。双分支、膨胀卷积和拼接会产生更多核函数启动和内存访问，因此理论运算量相同不代表实际速度相同。CBAM只增加0.79%的参数和0.08%的MACs，但延迟达到12.607毫秒，说明小模型的串行注意力算子难以仅用MACs评价。",
)
add_figure(doc, OUTPUTS / "figures" / "efficiency.png", "图 3 推理延迟与训练时间对比", width=6.55)

add_heading(doc, "4.3 类别错误分析", 2)
add_body(
    doc,
    "基线最难识别的类别是类别27行人标志，60张测试图像中仅正确31张，召回率51.67%。其次是类别22颠簸路面、类别30冰雪路面和类别6解除80公里限速。组合模型的行人类别召回率为50.00%，同时类别42解除大型车辆禁止超车的召回率降到77.78%。这些类别的测试样本较少，图案也与邻近警告或解除限制类别相似，导致Macro F1低于总体准确率。",
)
add_figure(doc, OUTPUTS / "baseline" / "confusion_matrix.png", "图 4 基线模型归一化混淆矩阵", width=5.9)

add_heading(doc, "4.4 场景鲁棒性", 2)
robustness_rows = []
for name in ("baseline", "combined"):
    row = robustness[name]
    robustness_rows.append(
        [
            DISPLAY_NAMES[name],
            f"{100 * row['clean']:.3f}",
            f"{100 * row['blur']:.3f}",
            f"{100 * row['dark']:.3f}",
            f"{100 * row['noise']:.3f}",
            f"{100 * row['mean_corrupted_accuracy']:.3f}",
        ]
    )
add_table(
    doc,
    ["模型", "干净", "模糊", "暗光", "噪声", "扰动均值"],
    robustness_rows,
    caption="表 5 固定扰动下的测试准确率（单位：%）",
    widths=[1.7, 0.78, 0.78, 0.78, 0.78, 1.0],
    font_size=8.5,
)
add_body(
    doc,
    "基线在模糊和暗光条件下仍保持95%以上准确率，但噪声准确率降到85.622%。组合模型三种扰动均低于基线，扰动平均准确率为91.848%，比基线低0.879个百分点。现有增强覆盖亮度和几何变化，却没有显式加入噪声训练，因此噪声成为最明显的失效场景。",
)
add_figure(doc, OUTPUTS / "figures" / "robustness.png", "图 5 基线与组合模型的场景鲁棒性", width=6.15)

add_heading(doc, "5 场景应用与可解释性", 1)
add_heading(doc, "5.1 单图交通标志识别", 2)
add_body(
    doc,
    "项目提供单图预测脚本，输出类别编号、英文类别名称、概率和Top 5候选。部署时建议使用基线权重，因为它在干净测试、三种扰动和实际延迟上均优于组合模型。对于低置信度输入，可以显示Top 5并交由人工确认，避免将分类结果直接作为车辆控制指令。",
)
add_code(doc, "python predict.py image.png --checkpoint outputs\\baseline\\best.pt --top-k 5")

add_heading(doc, "5.2 Grad-CAM可解释性", 2)
add_body(
    doc,
    "Grad CAM使用第四残差阶段最后一个归一化层的特征图和目标类别梯度生成热力图。停止标志示例中，基线的高响应区域集中在红色八边形主体及STOP文字附近，而背景响应较低，说明预测主要依赖标志本身。可视化不能证明因果正确性，但能辅助检查模型是否关注无关背景。",
)
add_figure(doc, OUTPUTS / "figures" / "gradcam_baseline_stop.png", "图 6 基线模型对停止标志的Grad-CAM结果", width=6.55)

add_heading(doc, "6 复现方法与代码仓库", 1)
add_heading(doc, "6.1 工程结构", 2)
add_body(
    doc,
    "工程将模型、数据、训练、评估和报告结果分离。每个实验目录包含config.json、history.csv、best.pt、metrics.json、训练曲线和混淆矩阵。报告表格由metrics.json自动读取，避免手工转录产生不一致。",
)
for line in (
    "gtsrb_resnet_cbam",
    "  src  models.py  data.py  utils.py  labels.py",
    "  train.py  robustness.py  predict.py  gradcam.py",
    "  aggregate_results.py  plot_results.py",
    "  outputs  各实验权重 指标 CSV 图表",
    "  reports  大作业报告",
    "  README.md  requirements.txt  download_gtsrb.ps1",
):
    add_code(doc, line)

add_heading(doc, "6.2 环境与运行步骤", 2)
add_body(doc, "在Windows PowerShell中可按以下顺序复现实验。CUDA轮子地址需要与本机驱动兼容。", first_indent=False)
commands = (
    "conda create -n gtsrb python=3.10 -y",
    "conda activate gtsrb",
    "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124",
    "pip install numpy matplotlib pillow tqdm pyyaml python-docx",
    "powershell -ExecutionPolicy Bypass -File .\\download_gtsrb.ps1",
    "python train.py --variant baseline --epochs 30 --batch-size 128 --lr 0.001",
    "powershell -ExecutionPolicy Bypass -File .\\run_ablation.ps1",
    "python train.py --variant combined --epochs 30 --batch-size 128 --lr 0.001",
    "python aggregate_results.py",
    "python robustness.py --checkpoint outputs\\baseline\\best.pt",
)
for command in commands:
    add_code(doc, command)

add_heading(doc, "6.3 仓库信息", 2)
add_body(
    doc,
    "项目已配置以下GitHub远程仓库地址，提交前应确认main分支能够在线访问。数据集、训练权重和中间图表已在.gitignore中排除，但experiment_summary.csv与experiment_summary.md会随仓库保留，便于核对报告汇总指标；大体积权重可通过Release或网盘提供。",
)
add_code(doc, "https://github.com/3037878831-cell/gtsrb-resnet18-ablation")

add_heading(doc, "7 局限与后续工作", 1)
add_body(
    doc,
    "本实验只使用一个随机种子，单次验证峰值可能包含抽样和优化波动。下一步应至少使用三个随机种子报告均值和标准差。GTSRB固定验证集与测试集存在可观察的分布差异，组合模型的验证优势没有迁移到测试集，因此还应采用交叉验证或额外道路采集数据。噪声准确率下降最明显，可以在不改变测试协议的前提下，在训练增强中加入不同强度的传感器噪声、JPEG压缩和运动模糊，再验证是否真正改善场景鲁棒性。",
)
add_body(
    doc,
    "最终部署还需要评估置信度校准、未知类别拒识和连续视频稳定性。当前系统是43类闭集分类器，不能判断输入是否属于数据集外交通标志，也不应直接连接车辆控制系统。实际应用应在检测器裁剪标志后进行分类，并为低置信度和异常输入设置人工或规则回退。",
)

add_heading(doc, "参考文献", 1)
references = (
    "[1] Stallkamp J, Schlipsing M, Salmen J, Igel C. The German Traffic Sign Recognition Benchmark. IJCNN, 2011.",
    "[2] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.",
    "[3] Woo S, Park J, Lee J Y, Kweon I S. CBAM Convolutional Block Attention Module. ECCV, 2018.",
    "[4] Wu Y, He K. Group Normalization. ECCV, 2018.",
    "[5] Misra D. Mish A Self Regularized Non Monotonic Neural Activation Function. BMVC, 2020.",
    "[6] PyTorch Documentation. https://pytorch.org/docs/stable/",
    "[7] Torchvision GTSRB Dataset. https://pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html",
)
for reference in references:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Pt(18)
    paragraph.paragraph_format.first_line_indent = Pt(-18)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.15
    set_run_font(paragraph.add_run(reference), size=9.5)

# Keep pictures with captions and avoid orphan headings.
for paragraph in doc.paragraphs:
    if paragraph.style and paragraph.style.name.startswith("Heading"):
        paragraph.paragraph_format.keep_with_next = True
    if paragraph.text.startswith("图 ") or paragraph.text.startswith("表 "):
        paragraph.paragraph_format.keep_with_next = False

doc.core_properties.title = "基于多尺度卷积与注意力机制的ResNet 18交通标志识别实验报告"
doc.core_properties.subject = "深度学习课程大作业"
doc.core_properties.author = ""
doc.core_properties.keywords = "GTSRB ResNet CBAM Mish"
doc.save(REPORT_PATH)
print(REPORT_PATH)

