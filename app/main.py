import streamlit as st
from PIL import Image, ImageOps
import io
from cartoonizer import AnimeGANv2Cartoonizer

# --- Page setup ---
st.set_page_config(
    page_title="Cartoonify Your Image",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Cartoonify Your Image")
st.write("Upload a photo and convert it into a cute cartoon version!")

# --- Параметры отображения ---
IMAGE_SIZE = (400, 400)

def fit_image(img, size=IMAGE_SIZE):
    """Обрезка и масштабирование изображения до нужного размера"""
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)

# --- Стиль и alpha ---
STYLES = ["face_paint", "hayao", "paprika"]
selected_style = st.selectbox("Choose cartoon style:", STYLES)
alpha = st.slider("Cartoon intensity", 0.0, 1.0, 1.0)

# --- Загрузка изображения ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Converting to cartoon..."):
        cartoonizer = AnimeGANv2Cartoonizer()
        cartoonizer.load(style=selected_style)  # загружаем выбранный стиль
        cartoon_image = cartoonizer.cartoonify(image)

        # Смешивание с оригиналом по alpha
        if alpha < 1.0:
            # Для blend
            blend_input = image.resize(cartoon_image.size)
            cartoon_image = Image.blend(blend_input, cartoon_image, alpha)
            #cartoon_image = Image.blend(image, cartoon_image, alpha)

    # --- Отображение изображений горизонтально ---
    st.subheader("Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(fit_image(image), caption="Original", use_container_width=True)
    with col2:
        st.image(fit_image(cartoon_image), caption=f"Cartoon ({selected_style})", use_container_width=True)

    st.success("Conversion completed! 😺")

    # --- Кнопка для скачивания ---
    buf = io.BytesIO()
    cartoon_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Download Cartoon Image",
        data=byte_im,
        file_name=f"cartoon_image_{selected_style}.png",
        mime="image/png"
    )
else:
    st.info("Please upload an image to see the cartoon version.")
