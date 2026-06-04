import time

import streamlit as st

from src.compress import compress_student_model
from src.foreground import fit_foreground_to_background, remove_background, resize_foreground
from src.image_preprocess import resize_by_long_edge
from src.infer import rank_candidates
from src.reference_opa import ensure_simopa_weight
from src.train import train_teacher_student


st.set_page_config(page_title="方向A-物体放置助手", layout="wide")
st.title("方向 A：智能物体放置与质量评分（本地推理）")
st.caption("默认使用 BCMI/libcom 的 SimOPA 参考模型进行位置评分（可切换到 legacy 对照）。")

with st.sidebar:
    st.header("模型与权重")
    if st.button("下载/检查 SimOPA 参考权重"):
        with st.spinner("准备 SimOPA 权重中..."):
            path = ensure_simopa_weight()
            st.success(f"已就绪: {path}")
    st.caption("建议默认使用 SimOPA（参考仓库预训练模型）。")
    model_backend = st.selectbox(
        "评分模型后端",
        options=["simopa", "legacy"],
        index=0,
        help="simopa: BCMI/libcom OPA 预训练模型；legacy: 当前仓库旧版模型",
    )
    if st.button("1) 离线训练教师+学生模型（高质量，CPU较久）"):
        with st.spinner("训练中，可能需要数分钟，请耐心等待..."):
            t0 = time.time()
            res = train_teacher_student(device="cpu")
            st.success(
                f"完成。teacher_loss={res.teacher_loss:.4f}, student_loss={res.student_loss:.4f}, "
                f"val_mae={res.student_val_mae:.4f}, val_corr={res.student_val_corr:.4f}, "
                f"耗时={time.time() - t0:.1f}s"
            )
    if st.button("2) 压缩学生模型（剪枝+量化）"):
        with st.spinner("压缩中..."):
            t0 = time.time()
            src_mb, dst_mb = compress_student_model()
            st.success(f"完成。{src_mb:.2f}MB -> {dst_mb:.2f}MB, 耗时={time.time() - t0:.1f}s")
    st.info("若使用 simopa，无需训练；若使用 legacy，建议先训练再压缩。")

bg_file = st.file_uploader(
    "上传背景图 (支持 jpg/jpeg/png/webp/bmp)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    key="bg",
)
fg_file = st.file_uploader("上传前景图 (支持 jpg/jpeg/png/webp/bmp)", type=["jpg", "jpeg", "png", "webp", "bmp"], key="fg")

st.subheader("前景处理选项")
col_a, col_b, col_c = st.columns(3)
with col_a:
    auto_cutout = st.checkbox("自动扣除背景", value=False)
with col_b:
    cutout_target = st.selectbox("抠图目标", options=["person", "foreground"], index=0)
with col_c:
    fg_scale = st.slider("前景缩放比例", min_value=0.3, max_value=2.5, value=1.0, step=0.05)

st.subheader("推理预处理")
resolution_profile = st.selectbox(
    "大图自动压缩策略（按长边）",
    options=["1080P (1920)", "2K (2560)", "关闭"],
    index=0,
    help="输入分辨率过大时，先缩小再推理以显著加速。",
)

use_compressed = st.checkbox("legacy 模型使用压缩权重（更快，精度略低）", value=False)

top_k = st.slider("展示推荐 Top-K", min_value=1, max_value=8, value=5)
candidate_count = st.slider("候选位置数量", min_value=6, max_value=20, value=12)

if st.button("开始推荐"):
    if bg_file is None or fg_file is None:
        st.warning("请先上传背景图和前景图。")
    else:
        from PIL import Image

        bg = Image.open(bg_file).convert("RGB")
        orig_bg_size = bg.size
        if resolution_profile.startswith("1080P"):
            bg = resize_by_long_edge(bg, 1920)
        elif resolution_profile.startswith("2K"):
            bg = resize_by_long_edge(bg, 2560)
        resized_bg_size = bg.size

        raw_fg = Image.open(fg_file)
        fg_info = "使用原图 alpha 通道（或不透明前景）。"
        if auto_cutout:
            fg, fg_info = remove_background(raw_fg, target=cutout_target)
        else:
            fg = raw_fg.convert("RGBA")
        fg = resize_foreground(fg, fg_scale)
        fg = fit_foreground_to_background(fg, bg)

        st.caption(fg_info)
        if resized_bg_size != orig_bg_size:
            st.caption(f"背景图已预缩放: {orig_bg_size[0]}x{orig_bg_size[1]} -> {resized_bg_size[0]}x{resized_bg_size[1]}")
        else:
            st.caption(f"背景图尺寸保持不变: {orig_bg_size[0]}x{orig_bg_size[1]}")
        preview_cols = st.columns(2)
        with preview_cols[0]:
            st.image(bg, caption="背景图", use_container_width=True)
        with preview_cols[1]:
            st.image(fg, caption="处理后前景图", use_container_width=True)

        with st.spinner("本地推理中..."):
            t0 = time.time()
            try:
                ranked, images = rank_candidates(
                    bg,
                    fg,
                    top_k=top_k,
                    candidate_count=candidate_count,
                    prefer_compressed=use_compressed,
                    model_backend=model_backend,
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            latency_ms = (time.time() - t0) * 1000.0

        st.success(f"完成。总耗时 {latency_ms:.1f} ms")
        cols = st.columns(len(images))
        for i, (row, img) in enumerate(zip(ranked, images)):
            with cols[i]:
                st.image(img, caption=f"#{i+1} 分数={row['score']:.3f} ({row['level']})", use_container_width=True)
                st.write(f"位置: x={row['x']}, y={row['y']}")

        st.subheader("排序结果")
        st.dataframe(ranked, use_container_width=True)

st.markdown("---")
st.write("运行入口: `streamlit run app.py`")
