import argparse
import tempfile
import zipfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


NAVY = RGBColor(0x05, 0x0A, 0xA0)
BLUE = RGBColor(0x05, 0x35, 0xE8)
MID_BLUE = RGBColor(0x2F, 0x75, 0xB5)
LIGHT_BLUE = RGBColor(0x5B, 0x9B, 0xD5)
PALE_BLUE = RGBColor(0xF5, 0xF7, 0xFA)
PALE_CYAN = RGBColor(0xF5, 0xF7, 0xFA)
PALE_YELLOW = RGBColor(0xFF, 0xFF, 0xFF)
PALE_RED = RGBColor(0xF5, 0xF7, 0xFA)
RED = RGBColor(0xE6, 0x00, 0x00)
ORANGE = RGBColor(0x2F, 0x75, 0xB5)
GREEN = RGBColor(0x2F, 0x75, 0xB5)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
FONT_NAME = "Microsoft YaHei"


def add_text_box(
    slide,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    font_size: float = 18,
    color: RGBColor = BLACK,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    vertical_anchor: MSO_ANCHOR = MSO_ANCHOR.MIDDLE,
    fill: RGBColor | None = None,
    line: RGBColor | None = None,
    margin: float = 0.08,
    radius: bool = False,
):
    shape_type = MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill if fill is not None else WHITE
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.2)

    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = vertical_anchor
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    run.font.name = FONT_NAME
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def add_rich_text_box(
    slide,
    segments: list[tuple[str, RGBColor, bool]],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    font_size: float,
    fill: RGBColor | None = None,
    line: RGBColor | None = None,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    margin: float = 0.12,
):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill if fill is not None else WHITE
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1.2)

    frame = shape.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_after = Pt(0)
    for text, color, bold in segments:
        run = paragraph.add_run()
        run.text = text
        run.font.name = FONT_NAME
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.color.rgb = color
    return shape


def add_header(slide, innovation_index: int, subtitle: str) -> None:
    add_text_box(
        slide,
        "三、项目主要创新成果",
        0.88,
        0.05,
        11.58,
        0.58,
        font_size=28,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.0),
        Inches(0.67),
        Inches(13.33),
        Inches(0.045),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = LIGHT_BLUE
    line.line.fill.background()
    add_rich_text_box(
        slide,
        [
            (f"■ 创新性成果{innovation_index}：", RED, True),
            (subtitle, BLUE, True),
        ],
        0.25,
        0.78,
        12.83,
        0.56,
        font_size=23,
    )


def add_chevron(slide, x: float, y: float, width: float = 0.38, height: float = 0.58) -> None:
    shape = slide.shapes.add_shape(
        MSO_SHAPE.CHEVRON,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_BLUE
    shape.line.fill.background()


def add_metric_card(
    slide,
    value: str,
    label: str,
    x: float,
    y: float,
    width: float,
    *,
    color: RGBColor,
) -> None:
    card = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(1.18),
    )
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = color
    card.line.width = Pt(2)
    frame = card.text_frame
    frame.clear()
    frame.margin_top = Inches(0.08)
    frame.margin_bottom = Inches(0.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_value = frame.paragraphs[0]
    p_value.alignment = PP_ALIGN.CENTER
    value_run = p_value.add_run()
    value_run.text = value
    value_run.font.name = FONT_NAME
    value_run.font.size = Pt(29)
    value_run.font.bold = True
    value_run.font.color.rgb = color
    p_label = frame.add_paragraph()
    p_label.alignment = PP_ALIGN.CENTER
    p_label.space_before = Pt(1)
    label_run = p_label.add_run()
    label_run.text = label
    label_run.font.name = FONT_NAME
    label_run.font.size = Pt(12)
    label_run.font.color.rgb = GRAY


def add_picture_contain(slide, path: Path, x: float, y: float, width: float, height: float):
    with Image.open(path) as image:
        image_width, image_height = image.size
    image_ratio = image_width / image_height
    box_ratio = width / height
    if image_ratio >= box_ratio:
        rendered_width = width
        rendered_height = width / image_ratio
        rendered_x = x
        rendered_y = y + (height - rendered_height) / 2
    else:
        rendered_height = height
        rendered_width = height * image_ratio
        rendered_x = x + (width - rendered_width) / 2
        rendered_y = y
    return slide.shapes.add_picture(
        str(path),
        Inches(rendered_x),
        Inches(rendered_y),
        width=Inches(rendered_width),
        height=Inches(rendered_height),
    )


def add_table(
    slide,
    rows: list[list[str]],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    emphasized_columns: set[int] | None = None,
) -> None:
    emphasized_columns = emphasized_columns or set()
    table_shape = slide.shapes.add_table(
        len(rows),
        len(rows[0]),
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(height),
    )
    table = table_shape.table
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            cell = table.cell(row_index, column_index)
            cell.text = value
            cell.margin_left = Inches(0.03)
            cell.margin_right = Inches(0.03)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if row_index == 0:
                cell.fill.fore_color.rgb = MID_BLUE
            elif row_index % 2:
                cell.fill.fore_color.rgb = PALE_BLUE
            else:
                cell.fill.fore_color.rgb = WHITE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.CENTER
            run = paragraph.runs[0]
            run.font.name = FONT_NAME
            run.font.size = Pt(11.5 if row_index else 12)
            run.font.bold = row_index == 0 or column_index in emphasized_columns
            run.font.color.rgb = WHITE if row_index == 0 else (RED if column_index in emphasized_columns else BLACK)


def add_config_chip(slide, text: str, x: float, y: float, width: float) -> None:
    add_text_box(
        slide,
        text,
        x,
        y,
        width,
        0.48,
        font_size=12,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=WHITE,
        line=LIGHT_BLUE,
        radius=True,
    )


def add_innovation_one_detail(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 1, "多源数据融合流程与关键设计")
    add_text_box(
        slide,
        "将不同时间频率、不同通道数的观测转换为统一时空序列，作为模型训练、验证和业务输出的数据基础。",
        0.45,
        1.38,
        12.43,
        0.58,
        font_size=16,
        color=BLACK,
        bold=True,
        fill=PALE_YELLOW,
        line=ORANGE,
        radius=True,
    )

    source_specs = [
        ("雷达", "1通道｜6 min", 2.05, MID_BLUE),
        ("卫星", "10通道｜10 min", 3.1, ORANGE),
        ("地面雨量", "1通道｜1 min", 4.05, GREEN),
    ]
    for name, detail, y, color in source_specs:
        add_text_box(
            slide,
            f"{name}\n{detail}",
            0.55,
            y,
            2.25,
            0.92,
            font_size=16,
            color=color,
            bold=True,
            align=PP_ALIGN.CENTER,
            fill=WHITE,
            line=color,
            radius=True,
        )

    add_chevron(slide, 3.0, 3.04, 0.45, 0.72)
    add_text_box(
        slide,
        "时空统一处理\n\n时间对齐\n空间配准\n归一化与裁剪\n降雨样本筛选",
        3.55,
        2.13,
        2.35,
        2.48,
        font_size=16,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_BLUE,
        line=LIGHT_BLUE,
        radius=True,
    )
    add_chevron(slide, 6.02, 3.04, 0.45, 0.72)
    add_text_box(
        slide,
        "统一多源序列\n\n12通道联合输入\n10/30 min时间尺度\n跨年度训练与验证",
        6.57,
        2.13,
        2.35,
        2.48,
        font_size=16,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_CYAN,
        line=MID_BLUE,
        radius=True,
    )
    add_chevron(slide, 9.05, 3.04, 0.45, 0.72)
    add_text_box(
        slide,
        "融合预测输出\n\n降雨为主要目标\n雷达/卫星辅助监督\n统一多阈值评测",
        9.6,
        2.13,
        3.05,
        2.48,
        font_size=16,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_RED,
        line=RED,
        radius=True,
    )

    add_text_box(
        slide,
        "关键设计",
        0.55,
        5.03,
        1.2,
        0.42,
        font_size=15,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=MID_BLUE,
        radius=True,
    )
    add_config_chip(slide, "输入：雷达1 + 卫星10 + 雨量1", 1.9, 5.0, 3.0)
    add_config_chip(slide, "统一切片：448 × 448", 5.05, 5.0, 2.2)
    add_config_chip(slide, "降雨区域筛选", 7.4, 5.0, 1.8)
    add_config_chip(slide, "CSI / POD / FAR / F1 / HSS", 9.35, 5.0, 3.2)
    add_text_box(
        slide,
        "该流程统一了三类观测的数据接口和评测口径，长时序预测与缺失模态实验均在此基础上开展。",
        0.55,
        5.78,
        12.1,
        0.72,
        font_size=17,
        color=BLACK,
        bold=True,
        fill=LIGHT_GRAY,
        line=LIGHT_BLUE,
        radius=True,
    )
    return slide


def add_innovation_one_metrics(prs: Presentation, report_table: Path, report_cover: Path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 1, "考核指标与第三方测试结果")
    add_rich_text_box(
        slide,
        [
            ("第三方测试表明：网格空间分辨率", BLACK, True),
            ("不低于1 km", RED, True),
            ("、产品更新频率", BLACK, True),
            ("不低于5 min", RED, True),
            ("，相关功能测试结论均为通过。", BLACK, True),
        ],
        0.42,
        1.38,
        12.48,
        0.68,
        font_size=18,
        fill=PALE_YELLOW,
        line=ORANGE,
    )
    add_metric_card(slide, "≥ 1 km", "网格空间分辨率", 0.58, 2.35, 1.75, color=RED)
    add_metric_card(slide, "≥ 5 min", "产品更新频率", 2.55, 2.35, 1.75, color=RED)
    add_metric_card(slide, "通过", "第三方功能测试", 4.52, 2.35, 1.75, color=MID_BLUE)
    add_text_box(
        slide,
        "支撑材料\n\n• 多源降雨数据融合预测算法1.0测试报告\n• 网格空间分辨率测试：通过\n• 更新频率测试：通过",
        0.58,
        3.85,
        5.7,
        1.92,
        font_size=16,
        color=BLACK,
        bold=True,
        fill=PALE_BLUE,
        line=LIGHT_BLUE,
        radius=True,
    )
    add_picture_contain(slide, report_table, 6.45, 2.08, 5.25, 4.55)
    add_picture_contain(slide, report_cover, 11.72, 3.42, 1.0, 1.45)
    add_text_box(
        slide,
        "第三方测试报告节选",
        7.95,
        6.43,
        3.25,
        0.35,
        font_size=11,
        color=GRAY,
        align=PP_ALIGN.CENTER,
    )
    return slide


def add_innovation_two_detail(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 2, "因果自回归长时序预测网络与训练细节")
    add_text_box(
        slide,
        "多源序列经Patch编码后依次执行空间注意力和因果时间注意力，再由三个3D解码分支恢复雷达、卫星和降雨序列。",
        0.45,
        1.38,
        12.43,
        0.58,
        font_size=16,
        color=BLACK,
        bold=True,
        fill=PALE_YELLOW,
        line=ORANGE,
        radius=True,
    )

    stages = [
        ("多源历史序列", "4帧 × 12通道\n30 min间隔", 0.42, 1.78, MID_BLUE),
        ("Patch编码", "8 × 8空间块\n映射至384维", 2.62, 1.78, GREEN),
        ("6×因果时空块", "空间注意力\n因果时间注意力\nMLP", 4.82, 2.25, ORANGE),
        ("3D多分支解码", "雷达1通道\n卫星10通道\n降雨1通道", 7.49, 2.1, MID_BLUE),
        ("下一帧/滚动预测", "序列监督\n4帧rollout分支", 10.02, 2.63, RED),
    ]
    for index, (title, detail, x, width, color) in enumerate(stages):
        add_text_box(
            slide,
            f"{title}\n\n{detail}",
            x,
            2.28,
            width,
            2.18,
            font_size=15,
            color=color,
            bold=True,
            align=PP_ALIGN.CENTER,
            fill=WHITE if index % 2 == 0 else PALE_BLUE,
            line=color,
            radius=True,
        )
        if index < len(stages) - 1:
            next_x = stages[index + 1][2]
            add_chevron(slide, x + width + 0.08, 3.02, max(next_x - (x + width) - 0.17, 0.28), 0.62)

    add_text_box(
        slide,
        "因果时序块",
        5.32,
        2.58,
        1.25,
        0.34,
        font_size=12,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=ORANGE,
        radius=True,
    )
    add_config_chip(slide, "模型规模：44.16M参数", 0.55, 4.88, 2.3)
    add_config_chip(slide, "Depth=6｜Heads=8｜Dim=384", 3.0, 4.88, 2.8)
    add_config_chip(slide, "2D空间RoPE + 1D时间RoPE", 5.95, 4.88, 2.9)
    add_config_chip(slide, "降雨残差学习", 9.0, 4.88, 1.65)
    add_config_chip(slide, "BF16训练", 10.8, 4.88, 1.65)
    add_text_box(
        slide,
        "训练采用雷达、卫星和降雨的联合下一帧序列监督，并加入权重为0.5的降雨滚动预测损失。",
        0.55,
        5.62,
        12.0,
        0.73,
        font_size=17,
        color=BLACK,
        bold=True,
        fill=LIGHT_GRAY,
        line=LIGHT_BLUE,
        radius=True,
    )
    add_text_box(
        slide,
        "说明：Wan-VAE latent-space路线作为独立探索实验，不属于本页最终Patch Transformer配置。",
        0.55,
        6.48,
        12.0,
        0.34,
        font_size=11,
        color=GRAY,
    )
    return slide


def add_innovation_two_metrics(prs: Presentation, rollout_gif: Path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 2, "长时序预测指标与动态视觉效果")
    add_rich_text_box(
        slide,
        [
            ("在", BLACK, True),
            ("30 min间隔、输入过去2 h", RED, True),
            ("的条件下，连续滚动预测未来", BLACK, True),
            ("3 h", RED, True),
            ("时，", BLACK, True),
            ("CSI@0.1达到0.5127", RED, True),
            ("。", BLACK, True),
        ],
        0.42,
        1.38,
        12.48,
        0.68,
        font_size=18,
        fill=PALE_YELLOW,
        line=ORANGE,
    )
    add_text_box(
        slide,
        "不同预测时效的CSI",
        0.52,
        2.15,
        5.55,
        0.42,
        font_size=16,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_table(
        slide,
        [
            ["预测时效", "CSI@0.01", "CSI@0.1", "CSI@0.2", "CSI@0.5"],
            ["30 min（t+1）", "0.7862", "0.7025", "0.5126", "0.2554"],
            ["1.5 h（t+3）", "0.7116", "0.6094", "0.4362", "0.1715"],
            ["3 h（t+6）", "0.6318", "0.5127", "0.3745", "0.1338"],
        ],
        0.48,
        2.62,
        5.65,
        2.45,
        emphasized_columns={2},
    )
    add_text_box(
        slide,
        "阈值单位：mm/min；CSI越高表示降雨区域命中效果越好。",
        0.55,
        5.15,
        5.55,
        0.55,
        font_size=12,
        color=GRAY,
        align=PP_ALIGN.CENTER,
        fill=LIGHT_GRAY,
        radius=True,
    )
    add_picture_contain(slide, rollout_gif, 6.38, 2.18, 6.45, 3.55)
    add_text_box(
        slide,
        "GIF动态播放｜GT与Pred逐时次对比",
        7.86,
        5.78,
        3.55,
        0.42,
        font_size=13,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=ORANGE,
        radius=True,
    )
    add_text_box(
        slide,
        "视觉结果显示：主要降雨带位置和空间结构能够随时间连续演化，强降雨核心区域保持较好的对应关系。",
        0.58,
        6.33,
        12.05,
        0.48,
        font_size=15,
        color=BLACK,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_BLUE,
        line=LIGHT_BLUE,
        radius=True,
    )
    return slide


def add_innovation_three_detail(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 3, "缺失模态统一预测机制与网络细节")
    add_text_box(
        slide,
        "训练阶段随机屏蔽部分观测模态，并使用可学习缺失标记替代相应输入，使同一模型能够处理不同模态组合。",
        0.45,
        1.38,
        12.43,
        0.58,
        font_size=16,
        color=BLACK,
        bold=True,
        fill=PALE_YELLOW,
        line=ORANGE,
        radius=True,
    )
    modality_specs = [
        ("雷达", "Drop 15%", 0.55, MID_BLUE),
        ("卫星", "Drop 15%", 2.55, ORANGE),
        ("雨量", "Drop 5%", 4.55, GREEN),
    ]
    for name, probability, x, color in modality_specs:
        add_text_box(
            slide,
            f"{name}\n{probability}",
            x,
            2.2,
            1.65,
            1.0,
            font_size=17,
            color=color,
            bold=True,
            align=PP_ALIGN.CENTER,
            fill=WHITE,
            line=color,
            radius=True,
        )
    add_chevron(slide, 6.43, 2.42, 0.48, 0.62)
    add_text_box(
        slide,
        "模态可用性采样\n至少保留1种可用模态",
        7.05,
        2.13,
        2.35,
        1.18,
        font_size=16,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_BLUE,
        line=LIGHT_BLUE,
        radius=True,
    )
    add_chevron(slide, 9.55, 2.42, 0.48, 0.62)
    add_text_box(
        slide,
        "可学习缺失标记\nMask Token",
        10.15,
        2.13,
        2.45,
        1.18,
        font_size=16,
        color=RED,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_RED,
        line=RED,
        radius=True,
    )

    add_text_box(
        slide,
        "完整模态",
        0.65,
        3.82,
        1.7,
        0.62,
        font_size=15,
        color=GREEN,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_CYAN,
        line=GREEN,
        radius=True,
    )
    add_text_box(
        slide,
        "缺失雷达/卫星/雨量",
        0.65,
        4.58,
        1.7,
        0.74,
        font_size=14,
        color=ORANGE,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_YELLOW,
        line=ORANGE,
        radius=True,
    )
    add_chevron(slide, 2.58, 4.17, 0.52, 0.72)
    add_text_box(
        slide,
        "共享因果Patch Transformer\n同一套参数处理不同输入组合",
        3.25,
        3.82,
        3.5,
        1.5,
        font_size=17,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_BLUE,
        line=MID_BLUE,
        radius=True,
    )
    add_chevron(slide, 6.95, 4.17, 0.52, 0.72)
    add_text_box(
        slide,
        "统一降雨预测输出\n无需为每种缺失组合单独训练",
        7.62,
        3.82,
        3.45,
        1.5,
        font_size=17,
        color=RED,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_RED,
        line=RED,
        radius=True,
    )
    add_text_box(
        slide,
        "共享模型\n参数",
        11.35,
        3.82,
        1.25,
        1.5,
        font_size=16,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=GREEN,
        radius=True,
    )
    add_text_box(
        slide,
        "当通信异常、设备离线或单一观测源暂时不可用时，模型可利用其余可用观测继续完成降雨预测。",
        0.58,
        5.83,
        12.05,
        0.72,
        font_size=17,
        color=BLACK,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=LIGHT_GRAY,
        line=LIGHT_BLUE,
        radius=True,
    )
    return slide


def add_innovation_three_metrics(prs: Presentation, missing_visual: Path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 3, "缺失模态预测指标与视觉效果")
    add_rich_text_box(
        slide,
        [
            ("在代表性缺失模态条件下，", BLACK, True),
            ("30 min间隔滚动预测未来3 h", RED, True),
            ("时，", BLACK, True),
            ("CSI@0.1达到0.4646", RED, True),
            ("，约保持完整模态结果的", BLACK, True),
            ("90.6%", RED, True),
            ("。", BLACK, True),
        ],
        0.42,
        1.38,
        12.48,
        0.68,
        font_size=18,
        fill=PALE_YELLOW,
        line=ORANGE,
    )
    add_text_box(
        slide,
        "完整模态与缺失模态对比（CSI@0.1）",
        0.48,
        2.18,
        5.75,
        0.42,
        font_size=16,
        color=NAVY,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_table(
        slide,
        [
            ["预测时效", "完整模态", "缺失模态", "性能保持率"],
            ["30 min（t+1）", "0.7025", "0.6768", "96.3%"],
            ["1.5 h（t+3）", "0.6094", "0.5697", "93.5%"],
            ["3 h（t+6）", "0.5127", "0.4646", "90.6%"],
        ],
        0.48,
        2.67,
        5.75,
        2.4,
        emphasized_columns={2, 3},
    )
    add_text_box(
        slide,
        "同一模型完成不同输入组合推理，避免为每种观测缺失情况分别训练和部署模型。",
        0.55,
        5.22,
        5.65,
        0.86,
        font_size=15,
        color=BLACK,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=PALE_BLUE,
        line=LIGHT_BLUE,
        radius=True,
    )
    add_picture_contain(slide, missing_visual, 6.45, 2.18, 6.35, 3.7)
    add_text_box(
        slide,
        "代表性缺失模态条件下的GT / Pred对比",
        7.75,
        5.92,
        3.7,
        0.42,
        font_size=13,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        fill=MID_BLUE,
        radius=True,
    )
    add_text_box(
        slide,
        "注：汇报时应补充该表对应的具体缺失模态组合，避免将代表性设置解释为所有组合的统一结果。",
        0.6,
        6.48,
        12.0,
        0.33,
        font_size=11,
        color=GRAY,
        align=PP_ALIGN.CENTER,
    )
    return slide


def extract_assets(source_pptx: Path, output_dir: Path) -> dict[str, Path]:
    asset_names = {
        "rollout_gif": "ppt/media/image82.GIF",
        "missing_visual": "ppt/media/image87.png",
        "report_table": "ppt/media/image96.png",
        "report_cover": "ppt/media/image97.png",
    }
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(source_pptx) as archive:
        for key, member in asset_names.items():
            destination = output_dir / Path(member).name
            destination.write_bytes(archive.read(member))
            extracted[key] = destination
    return extracted


def update_slide_numbers(prs: Presentation) -> None:
    for slide_number, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if "编号占位符" in shape.name and shape.has_text_frame:
                shape.text = str(slide_number)
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = FONT_NAME


def normalize_existing_content(prs: Presentation) -> None:
    summary_slide = prs.slides[13]
    for shape in summary_slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.text.startswith("经过两年的攻关"):
            runs = shape.text_frame.paragraphs[0].runs
            runs[1].text = "“"
            runs[3].text = "—"
            runs[5].text = "—"
            runs[7].text = "”"
            runs[9].text = (
                "，从多源异构观测协同建模、长时效预测误差累积抑制和观测中断适应三个方面，"
                "系统性解决了复杂山区降雨预测中多源数据难融合、长时预测不稳定及缺失模态难适配的难题。"
            )
        elif "创新性成果1：构建雷达" in shape.text:
            replacement_texts = {
                0: "：开发出多源降雨数据融合预测算法",
                2: "：开发出因果自回归长时序降雨预测模型",
                4: "：开发出缺失模态统一降雨预测模型",
            }
            for paragraph_index, replacement in replacement_texts.items():
                paragraph = shape.text_frame.paragraphs[paragraph_index]
                paragraph.runs[2].text = replacement
                for run in paragraph.runs[3:]:
                    run.text = ""

    long_sequence_slide = prs.slides[15]
    for shape in long_sequence_slide.shapes:
        if shape.has_text_frame and "输入过去任意时长" in shape.text:
            paragraph = shape.text_frame.paragraphs[2]
            paragraph.runs[0].text = "②支持输入过去2小时多源观测数据并连续滚动预测未来"

    missing_modality_slide = prs.slides[16]
    for shape in missing_modality_slide.shapes:
        if shape.has_text_frame and "输入序列每种模态至少保留一帧数据即可推理" in shape.text:
            paragraph = shape.text_frame.paragraphs[0]
            paragraph.runs[-1].text = "，提高了观测数据中断条件下的预测鲁棒性，已达到至少保留一种可用观测模态即可完成推理"


def build_presentation(input_path: Path, source_path: Path, output_path: Path) -> None:
    presentation = Presentation(input_path)
    original_slide_ids = list(presentation.slides._sldIdLst)
    if len(original_slide_ids) != 24:
        raise ValueError(f"Expected 24 slides in the edited template, got {len(original_slide_ids)}")
    normalize_existing_content(presentation)

    with tempfile.TemporaryDirectory(prefix="rainppt-assets-") as temp_directory:
        assets = extract_assets(source_path, Path(temp_directory))
        added_slides = [
            add_innovation_one_detail(presentation),
            add_innovation_one_metrics(presentation, assets["report_table"], assets["report_cover"]),
            add_innovation_two_detail(presentation),
            add_innovation_two_metrics(presentation, assets["rollout_gif"]),
            add_innovation_three_detail(presentation),
            add_innovation_three_metrics(presentation, assets["missing_visual"]),
        ]

        added_slide_ids = list(presentation.slides._sldIdLst)[-len(added_slides) :]
        ordered_slide_ids = []
        for original_index, slide_id in enumerate(original_slide_ids, start=1):
            ordered_slide_ids.append(slide_id)
            if original_index == 15:
                ordered_slide_ids.extend(added_slide_ids[0:2])
            elif original_index == 16:
                ordered_slide_ids.extend(added_slide_ids[2:4])
            elif original_index == 17:
                ordered_slide_ids.extend(added_slide_ids[4:6])

        slide_id_list = presentation.slides._sldIdLst
        for slide_id in list(slide_id_list):
            slide_id_list.remove(slide_id)
        for slide_id in ordered_slide_ids:
            slide_id_list.append(slide_id)

        update_slide_numbers(presentation)
        presentation.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_presentation(args.input, args.source, args.output)


if __name__ == "__main__":
    main()
