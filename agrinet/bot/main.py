from dotenv import load_dotenv
import requests
import os
import io
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from PIL import Image as PILImage, ImageFilter, ImageDraw
import matplotlib
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

load_dotenv()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")

IMAGE_WIDTH = IMAGE_HEIGHT = 512
RGBN_CHANNELS = 4
BACKBONE_MODEL = "efficientnet-b2"

LABELS = [
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "waterway",
    "water",
    "planter_skip",
    "weed_cluster",
]

EXCLUDE_LABELS = ["storm_damage"]
CLASSES_COUNT = len(LABELS)

THRESHOLD = 0.5

PALETTE_DICT = {
    "double_plant": (255, 200, 0),
    "drydown": (210, 105, 30),
    "endrow": (255, 165, 0),
    "nutrient_deficiency": (255, 0, 255),
    "waterway": (0, 191, 255),
    "water": (0, 128, 255),
    "planter_skip": (255, 20, 147),
    "weed_cluster": (0, 100, 0),
}

PALETTE = [PALETTE_DICT[label] for label in LABELS]

MODEL_PATH = f"best_model_{BACKBONE_MODEL}_dataset.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conversation states
WAITING_RGB = 0
WAITING_NIR = 1

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# Localization System
# ============================================================


class Locale:
    """Internationalization support for RU and EN."""

    CLASS_NAMES = {
        "en": {
            "double_plant": "Double Plant",
            "drydown": "Drydown",
            "endrow": "End Row",
            "nutrient_deficiency": "Nutrient Deficiency",
            "waterway": "Waterway",
            "water": "Water",
            "planter_skip": "Planter Skip",
            "weed_cluster": "Weed Cluster",
        },
        "ru": {
            "double_plant": "Двойная посадка",
            "drydown": "Высыхание",
            "endrow": "Конец ряда",
            "nutrient_deficiency": "Дефицит питательных в-в",
            "waterway": "Водоток",
            "water": "Вода",
            "planter_skip": "Пропуск сеялки",
            "weed_cluster": "Скопление сорняков",
        },
    }

    STRINGS = {
        "en": {
            "welcome_title": "🛰️ *Agricultural Anomaly Detection Bot*\n\n",
            "welcome_desc": (
                "This bot analyzes aerial/drone imagery to detect agricultural anomalies "
                "using a trained DeepLabV3+ neural network.\n\n"
            ),
            "detectable_classes": "📋 *Detectable Classes:*\n",
            "how_to_use": (
                "\n📸 *How to use:*\n"
                "1️⃣ Send an RGB image (regular photo)\n"
                "2️⃣ Then send the corresponding NIR (Near-Infrared) image\n\n"
                "The bot will return:\n"
                "• NDVI visualization\n"
                "• Prediction overlay with detected anomalies\n"
                "• List of detected classes with colors\n\n"
                "Send /analyze to start analysis or just send your RGB image!"
            ),
            "send_rgb": (
                "📸 *Step 1/2:* Please send the *RGB image* (regular color photo).\n\n"
                "💡 _Send as a file (document) for best quality, or as a photo._"
            ),
            "send_nir": (
                "✅ RGB image received! Size: {width}×{height}\n\n"
                "📸 *Step 2/2:* Now send the *NIR (Near-Infrared) image*.\n\n"
                "💡 _This should be the same scene captured in near-infrared._\n"
                "Or send /skip\\_nir to use grayscale as synthetic NIR (less accurate)."
            ),
            "send_image_error": "❌ Please send an image (as photo or document).",
            "processing": "⏳ Processing images... Please wait.",
            "rgb_not_found": "❌ RGB image not found. Please start over with /analyze",
            "ndvi_caption": "🗺️ NDVI Visualization",
            "ndvi_stats_title": "📊 *NDVI Statistics:*\n",
            "ndvi_min": "  • Min: `{value}`\n",
            "ndvi_max": "  • Max: `{value}`\n",
            "ndvi_mean": "  • Mean: `{value}`\n",
            "veg_health_title": "\n🌿 *Vegetation Health:*\n",
            "veg_healthy": "  • Healthy (NDVI > 0.3): `{value}%`\n",
            "veg_sparse": "  • Sparse (0.1–0.3): `{value}%`\n",
            "veg_nonveg": "  • Non-vegetation (≤ 0.1): `{value}%`",
            "prediction_caption": "🔍 Prediction Overlay (detected anomalies highlighted)",
            "legend_caption": "📋 Color Legend",
            "no_anomalies": "✅ No anomalies detected at current threshold.",
            "detected_title": "🔍 *Detected Anomalies:*\n\n",
            "detected_item": (
                "  {emoji} *{name}*\n"
                "      Color: `{hex}` RGB`{rgb}`\n"
                "      Pixels: `{pixels}` ({percent}%)\n\n"
            ),
            "analysis_complete": "✅ *Analysis Complete!*\n\n",
            "input_size": "📐 Input: {orig_w}×{orig_h} → {target_w}×{target_h}\n",
            "threshold_info": "🎯 Threshold: `{threshold}`\n",
            "classes_detected": "🏷️ Detected: `{count}/{total}` classes\n\n",
            "send_another": "Send another image or /analyze to start over.",
            "help_title": "🛰️ *Agricultural Anomaly Detection Bot*\n\n",
            "help_commands": (
                "*Commands:*\n"
                "/start — Welcome message and instructions\n"
                "/analyze — Start new image analysis\n"
                "/skip\\_nir — Use grayscale as synthetic NIR\n"
                "/threshold — Show/set prediction threshold\n"
                "/classes — Show all detectable classes\n"
                "/lang — Switch language (EN/RU)\n"
                "/help — This help message\n"
                "/cancel — Cancel current operation\n\n"
            ),
            "help_workflow": (
                "*Workflow:*\n"
                "1. Send RGB image\n"
                "2. Send NIR image (or /skip\\_nir)\n"
                "3. Receive analysis results\n\n"
            ),
            "help_info": (
                "*Supported formats:* JPG, PNG, TIFF\n"
                "*Model:* DeepLabV3+ with EfficientNet-B2\n"
                "*Input size:* {width}×{height}\n"
                "*Classes:* {classes}"
            ),
            "classes_title": "🏷️ *Detectable Anomaly Classes:*\n\n",
            "threshold_current": "🎯 Current threshold: `{threshold}`\n\nUsage: `/threshold 0.3` to change.",
            "threshold_updated": "✅ Threshold updated to `{threshold}`",
            "threshold_error_range": "❌ Threshold must be between 0 and 1.",
            "threshold_error_value": "❌ Invalid number. Usage: /threshold 0.5",
            "skip_nir_warning": (
                "⚠️ Using grayscale as synthetic NIR. Results may be less accurate.\n"
                "⏳ Processing..."
            ),
            "skip_nir_no_rgb": "❌ No RGB image found. Send RGB image first.",
            "cancelled": "❌ Operation cancelled. Send /analyze to start over.",
            "lang_switched": "🌐 Language set to *English*.",
            "lang_prompt": "🌐 Choose language / Выберите язык:\n/lang\\_en — English\n/lang\\_ru — Русский",
            "error_generic": "❌ An unexpected error occurred. Please try again with /analyze",
            "error_processing": "❌ Error processing image: {error}\nPlease try again.",
            "error_analysis": "❌ Analysis failed: {error}\nPlease try again with /analyze",
            "detected_4ch": "⏳ Detected 4-channel image. Processing as RGBN...",
            "legend_no_detections": "No anomalies detected",
            "legend_title": "Detected Classes:",
            "ndvi_plot_title": "NDVI",
            "ndvi_plot_stats": "min={min}, max={max}, mean={mean}",
            "ndvi_plot_healthy": "Healthy (NDVI > 0.3): {value}%",
            "ndvi_plot_sparse": "Sparse (0.1–0.3): {value}%",
            "ndvi_plot_nonveg": "Non-vegetation (≤ 0.1): {value}%",
        },
        "ru": {
            "welcome_title": "🛰️ *Бот обнаружения аномалий в сельском хозяйстве*\n\n",
            "welcome_desc": (
                "Этот бот анализирует аэро/дрон-снимки для обнаружения аномалий "
                "сельскохозяйственных угодий с помощью обученной нейросети DeepLabV3+.\n\n"
            ),
            "detectable_classes": "📋 *Обнаруживаемые классы:*\n",
            "how_to_use": (
                "\n📸 *Как использовать:*\n"
                "1️⃣ Отправьте RGB-снимок (обычное цветное фото)\n"
                "2️⃣ Затем отправьте соответствующий NIR-снимок (ближний инфракрасный)\n\n"
                "Бот вернёт:\n"
                "• Визуализацию NDVI\n"
                "• Наложение предсказания с обнаруженными аномалиями\n"
                "• Список обнаруженных классов с цветами\n\n"
                "Отправьте /analyze для начала анализа или просто отправьте RGB-снимок!"
            ),
            "send_rgb": (
                "📸 *Шаг 1/2:* Отправьте *RGB-снимок* (обычное цветное фото).\n\n"
                "💡 _Для лучшего качества отправьте как файл (документ) или как фото._"
            ),
            "send_nir": (
                "✅ RGB-снимок получен! Размер: {width}×{height}\n\n"
                "📸 *Шаг 2/2:* Теперь отправьте *NIR-снимок (ближний инфракрасный)*.\n\n"
                "💡 _Это должна быть та же сцена, снятая в ближнем ИК-диапазоне._\n"
                "Или отправьте /skip\\_nir для использования градаций серого как NIR (менее точно)."
            ),
            "send_image_error": "❌ Пожалуйста, отправьте изображение (как фото или документ).",
            "processing": "⏳ Обработка изображений... Пожалуйста, подождите.",
            "rgb_not_found": "❌ RGB-снимок не найден. Начните заново с /analyze",
            "ndvi_caption": "🗺️ Визуализация NDVI",
            "ndvi_stats_title": "📊 *Статистика NDVI:*\n",
            "ndvi_min": "  • Мин: `{value}`\n",
            "ndvi_max": "  • Макс: `{value}`\n",
            "ndvi_mean": "  • Среднее: `{value}`\n",
            "veg_health_title": "\n🌿 *Здоровье растительности:*\n",
            "veg_healthy": "  • Здоровая (NDVI > 0.3): `{value}%`\n",
            "veg_sparse": "  • Разреженная (0.1–0.3): `{value}%`\n",
            "veg_nonveg": "  • Без растительности (≤ 0.1): `{value}%`",
            "prediction_caption": "🔍 Наложение предсказания (обнаруженные аномалии выделены)",
            "legend_caption": "📋 Цветовая легенда",
            "no_anomalies": "✅ Аномалий не обнаружено при текущем пороге.",
            "detected_title": "🔍 *Обнаруженные аномалии:*\n\n",
            "detected_item": (
                "  {emoji} *{name}*\n"
                "      Цвет: `{hex}` RGB`{rgb}`\n"
                "      Пикселей: `{pixels}` ({percent}%)\n\n"
            ),
            "analysis_complete": "✅ *Анализ завершён!*\n\n",
            "input_size": "📐 Вход: {orig_w}×{orig_h} → {target_w}×{target_h}\n",
            "threshold_info": "🎯 Порог: `{threshold}`\n",
            "classes_detected": "🏷️ Обнаружено: `{count}/{total}` классов\n\n",
            "send_another": "Отправьте ещё изображение или /analyze для нового анализа.",
            "help_title": "🛰️ *Бот обнаружения аномалий в сельском хозяйстве*\n\n",
            "help_commands": (
                "*Команды:*\n"
                "/start — Приветствие и инструкции\n"
                "/analyze — Начать новый анализ\n"
                "/skip\\_nir — Использовать градации серого как NIR\n"
                "/threshold — Показать/установить порог предсказания\n"
                "/classes — Показать все обнаруживаемые классы\n"
                "/lang — Переключить язык (EN/RU)\n"
                "/help — Это сообщение помощи\n"
                "/cancel — Отменить текущую операцию\n\n"
            ),
            "help_workflow": (
                "*Порядок работы:*\n"
                "1. Отправьте RGB-снимок\n"
                "2. Отправьте NIR-снимок (или /skip\\_nir)\n"
                "3. Получите результаты анализа\n\n"
            ),
            "help_info": (
                "*Поддерживаемые форматы:* JPG, PNG, TIFF\n"
                "*Модель:* DeepLabV3+ с EfficientNet-B2\n"
                "*Размер входа:* {width}×{height}\n"
                "*Классов:* {classes}"
            ),
            "classes_title": "🏷️ *Обнаруживаемые классы аномалий:*\n\n",
            "threshold_current": "🎯 Текущий порог: `{threshold}`\n\nИспользование: `/threshold 0.3` для изменения.",
            "threshold_updated": "✅ Порог обновлён: `{threshold}`",
            "threshold_error_range": "❌ Порог должен быть от 0 до 1.",
            "threshold_error_value": "❌ Неверное число. Использование: /threshold 0.5",
            "skip_nir_warning": (
                "⚠️ Используются градации серого как синтетический NIR. Результаты могут быть менее точными.\n"
                "⏳ Обработка..."
            ),
            "skip_nir_no_rgb": "❌ RGB-снимок не найден. Сначала отправьте RGB-снимок.",
            "cancelled": "❌ Операция отменена. Отправьте /analyze для начала.",
            "lang_switched": "🌐 Язык установлен: *Русский*.",
            "lang_prompt": "🌐 Choose language / Выберите язык:\n/lang\\_en — English\n/lang\\_ru — Русский",
            "error_generic": "❌ Произошла ошибка. Попробуйте снова с /analyze",
            "error_processing": "❌ Ошибка обработки: {error}\nПопробуйте снова.",
            "error_analysis": "❌ Анализ не удался: {error}\nПопробуйте снова с /analyze",
            "detected_4ch": "⏳ Обнаружено 4-канальное изображение. Обработка как RGBN...",
            "legend_no_detections": "Аномалий не обнаружено",
            "legend_title": "Обнаруженные классы:",
            "ndvi_plot_title": "NDVI",
            "ndvi_plot_stats": "мин={min}, макс={max}, среднее={mean}",
            "ndvi_plot_healthy": "Здоровая (NDVI > 0.3): {value}%",
            "ndvi_plot_sparse": "Разреженная (0.1–0.3): {value}%",
            "ndvi_plot_nonveg": "Без растительности (≤ 0.1): {value}%",
        },
    }

    @classmethod
    def get_lang(cls, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        if context.user_data and context.user_data.get("lang"):
            return context.user_data["lang"]
        if update.effective_user and update.effective_user.language_code:
            lang_code = update.effective_user.language_code.lower()
            if lang_code.startswith("ru"):
                return "ru"
        # Default to Russian instead of English
        return "ru"

    @classmethod
    def t(cls, key: str, update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs) -> str:
        lang = cls.get_lang(update, context)
        # Use Russian as the primary fallback
        strings = cls.STRINGS.get(lang, cls.STRINGS["ru"])
        template = strings.get(key, cls.STRINGS["ru"].get(key, f"[{key}]"))
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError):
                return template
        return template

    @classmethod
    def get_class_name(cls, label: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        lang = cls.get_lang(update, context)
        # Fallback to Russian
        names = cls.CLASS_NAMES.get(lang, cls.CLASS_NAMES["ru"])
        return names.get(label, label)

    @classmethod
    def get_class_names_for_lang(cls, lang: str) -> dict:
        # Fallback to Russian
        return cls.CLASS_NAMES.get(lang, cls.CLASS_NAMES["ru"])


# ============================================================
# Model Loading
# ============================================================


def load_model() -> nn.Module:
    """Load the trained DeepLabV3Plus model."""
    model = smp.DeepLabV3Plus(
        encoder_name=BACKBONE_MODEL,
        encoder_weights=None,
        in_channels=RGBN_CHANNELS,
        classes=CLASSES_COUNT,
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please place the trained model in the current directory."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_miou = checkpoint.get("best_miou", "N/A")
    logger.info(f"Model loaded successfully. Best mIoU: {best_miou}")

    return model


# ============================================================
# Image Processing Utilities
# ============================================================


def preprocess_image(
    rgb_image: PILImage.Image, nir_image: PILImage.Image
) -> torch.Tensor:
    """
    Preprocess RGB + NIR images into a 4-channel tensor.

    Normalization matches training pipeline:
      - Divide by 255 (8-bit) or 65535 (16-bit) → [0, 1]
      - NO ImageNet mean/std normalization (model was trained without it)
      - Resize to 512×512
    """
    rgb_arr = np.array(rgb_image.convert("RGB")).astype(np.float32)
    nir_arr = np.array(nir_image).astype(np.float32)

    # Detect bit depth and normalize to [0, 1] — same as training
    rgb_scale = 65535.0 if rgb_arr.max() > 255 else 255.0
    nir_scale = 65535.0 if nir_arr.max() > 255 else 255.0

    rgb_arr = rgb_arr / rgb_scale
    nir_arr = nir_arr / nir_scale

    # Ensure NIR is 2D (H, W)
    if nir_arr.ndim == 3:
        nir_arr = nir_arr[..., 0]

    # Stack into RGBN (H, W, 4) → (4, H, W)
    rgbn = np.concatenate([rgb_arr, nir_arr[..., np.newaxis]], axis=-1)
    img_tensor = torch.from_numpy(rgbn).permute(2, 0, 1).contiguous()

    # Resize to model input size (matches training: all images are 512×512)
    img_tensor = TF.resize(
        img_tensor,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )

    return img_tensor


def calculate_ndvi(rgbn_tensor: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Calculate NDVI from RGBN tensor of shape (4, H, W) in [0, 1]."""
    arr = rgbn_tensor.cpu().numpy()
    red = arr[0].astype(np.float32)
    nir = arr[3].astype(np.float32)

    if red.max() > 1.5 or nir.max() > 1.5:
        red = red / 255.0
        nir = nir / 255.0

    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1.0, 1.0)


def generate_ndvi_image(rgbn_tensor: torch.Tensor, lang: str = "en") -> PILImage.Image:
    """Generate NDVI visualization as a PIL Image."""
    ndvi = calculate_ndvi(rgbn_tensor)
    strings = Locale.STRINGS.get(lang, Locale.STRINGS["ru"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1.0, vmax=1.0)

    title = strings["ndvi_plot_title"]
    stats = strings["ndvi_plot_stats"].format(
        min=f"{ndvi.min():.3f}",
        max=f"{ndvi.max():.3f}",
        mean=f"{ndvi.mean():.3f}",
    )
    ax.set_title(f"{title}\n{stats}", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDVI")

    veg_healthy = (ndvi > 0.3).sum() / ndvi.size * 100
    veg_sparse = ((ndvi > 0.1) & (ndvi <= 0.3)).sum() / ndvi.size * 100
    non_veg = (ndvi <= 0.1).sum() / ndvi.size * 100

    stats_text = (
        f"{strings['ndvi_plot_healthy'].format(value=f'{veg_healthy:.1f}')}\n"
        f"{strings['ndvi_plot_sparse'].format(value=f'{veg_sparse:.1f}')}\n"
        f"{strings['ndvi_plot_nonveg'].format(value=f'{non_veg:.1f}')}"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return PILImage.open(buf).copy()


@torch.no_grad()
def predict(
    model: nn.Module, rgbn_tensor: torch.Tensor, threshold: float = THRESHOLD
) -> torch.Tensor:
    """Run model prediction. Returns binary mask (C, H, W)."""
    model.eval()
    img_batch = rgbn_tensor.unsqueeze(0).to(DEVICE)
    logits = model(img_batch)
    probs = torch.sigmoid(logits)
    bin_mask = (probs > threshold)[0].cpu()
    return bin_mask


def generate_prediction_overlay(
    rgbn_tensor: torch.Tensor,
    pred_mask: torch.Tensor,
    alpha: float = 0.6,
) -> PILImage.Image:
    """Generate prediction overlay image."""
    rgb_np = rgbn_tensor[:3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    rgb_uint8 = (rgb_np * 255).astype(np.uint8)

    mask_np = pred_mask.cpu().numpy()
    C, H, W = mask_np.shape

    color_overlay = np.zeros((H, W, 3), dtype=np.float32)
    mask_present = np.zeros((H, W), dtype=bool)

    for c in range(C):
        mask_c = mask_np[c].astype(bool)
        if mask_c.any():
            mask_present |= mask_c
            color = np.array(PALETTE[c], dtype=np.float32)
            color_overlay[mask_c] += color

    color_overlay = np.clip(color_overlay, 0, 255).astype(np.uint8)

    blended = rgb_uint8.copy()
    if mask_present.any():
        blended[mask_present] = (
            (1.0 - alpha) * rgb_uint8[mask_present].astype(np.float32)
            + alpha * color_overlay[mask_present].astype(np.float32)
        ).astype(np.uint8)

    pil_img = PILImage.fromarray(blended).filter(ImageFilter.GaussianBlur(radius=0.7))
    return pil_img


def generate_legend_image(
    detected_classes: list[tuple[str, tuple, int]],
    lang: str = "en",
) -> PILImage.Image:
    """Generate a legend image showing detected classes."""
    strings = Locale.STRINGS.get(lang, Locale.STRINGS["ru"])
    class_names_dict = Locale.get_class_names_for_lang(lang)

    if not detected_classes:
        img = PILImage.new("RGB", (400, 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), strings["legend_no_detections"], fill=(0, 0, 0))
        return img

    line_height = 40
    padding = 20
    width = 550
    height = padding * 2 + line_height * len(detected_classes) + 30

    img = PILImage.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((padding, 10), strings["legend_title"], fill=(0, 0, 0))

    y_offset = 40
    for name, color, pixels in detected_classes:
        x = padding
        draw.rectangle(
            [x, y_offset + 5, x + 30, y_offset + 30],
            fill=color, outline=(0, 0, 0),
        )
        translated_name = class_names_dict.get(name, name)
        percentage = pixels / (IMAGE_WIDTH * IMAGE_HEIGHT) * 100
        text = f"{translated_name} — {pixels:,} px ({percentage:.1f}%)"
        draw.text((x + 40, y_offset + 8), text, fill=(0, 0, 0))
        y_offset += line_height

    return img


def _get_class_emoji(class_name: str) -> str:
    emoji_map = {
        "double_plant": "🌱🌱",
        "drydown": "🥀",
        "endrow": "🔚",
        "nutrient_deficiency": "⚠️",
        "waterway": "💧",
        "water": "🌊",
        "planter_skip": "⏭️",
        "weed_cluster": "🌿",
    }
    return emoji_map.get(class_name, "•")


def pil_to_bytes(img: PILImage.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Helper: Download image from Telegram message
# ============================================================


async def _download_image_from_message(update: Update) -> PILImage.Image | None:
    """
    Extract and download image from a Telegram message.
    Returns PIL Image or None if no image found.
    """
    if update.message.document and update.message.document.mime_type and \
       update.message.document.mime_type.startswith("image/"):
        file = await update.message.document.get_file()
    elif update.message.photo:
        # Get highest resolution
        file = await update.message.photo[-1].get_file()
    else:
        return None

    file_bytes = await file.download_as_bytearray()
    return PILImage.open(io.BytesIO(file_bytes))


# ============================================================
# Telegram Bot Handlers
# ============================================================

model: nn.Module = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /start command."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    text = t("welcome_title")
    text += t("welcome_desc")
    text += t("detectable_classes")

    for label in LABELS:
        color = PALETTE_DICT[label]
        emoji = _get_class_emoji(label)
        color_hex = "#{:02x}{:02x}{:02x}".format(*color)
        class_name = Locale.get_class_name(label, update, context)
        text += f"  {emoji} {class_name} — `{color_hex}`\n"

    text += t("how_to_use")

    await update.message.reply_text(text, parse_mode="Markdown")
    return WAITING_RGB


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /analyze command."""
    lang = context.user_data.get("lang")
    context.user_data.clear()
    if lang:
        context.user_data["lang"] = lang

    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    await update.message.reply_text(t("send_rgb"), parse_mode="Markdown")
    return WAITING_RGB


async def receive_rgb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive and store RGB image. Handles both first-image and explicit RGB step."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    try:
        image = await _download_image_from_message(update)
        if image is None:
            await update.message.reply_text(t("send_image_error"))
            return WAITING_RGB

        img_arr = np.array(image)

        # Check if 4-channel PNG (RGBN) — process immediately
        if img_arr.ndim == 3 and img_arr.shape[2] == 4:
            await update.message.reply_text(t("detected_4ch"))

            rgbn_arr = img_arr.astype(np.float32)
            scale = 65535.0 if rgbn_arr.max() > 255 else 255.0
            rgbn_arr = rgbn_arr / scale

            rgbn_tensor = torch.from_numpy(rgbn_arr).permute(2, 0, 1).contiguous()
            rgbn_tensor = TF.resize(
                rgbn_tensor,
                [IMAGE_HEIGHT, IMAGE_WIDTH],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            orig_size = (img_arr.shape[1], img_arr.shape[0])
            return await _run_analysis(update, context, rgbn_tensor, orig_size)

        # Regular image — store as RGB and wait for NIR
        rgb_image = image.convert("RGB")
        context.user_data["rgb_image"] = rgb_image

        logger.info(
            f"RGB image stored for user {update.effective_user.id}, "
            f"size={rgb_image.size}, transitioning to WAITING_NIR"
        )

        await update.message.reply_text(
            t("send_nir", width=rgb_image.size[0], height=rgb_image.size[1]),
            parse_mode="Markdown",
        )
        return WAITING_NIR

    except Exception as e:
        logger.error(f"Error receiving RGB image: {e}", exc_info=True)
        await update.message.reply_text(t("error_processing", error=str(e)))
        return WAITING_RGB


async def receive_nir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive NIR image and run full analysis."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    logger.info(f"receive_nir called for user {update.effective_user.id}")

    try:
        image = await _download_image_from_message(update)
        if image is None:
            await update.message.reply_text(t("send_image_error"))
            return WAITING_NIR

        # Get stored RGB image
        rgb_image = context.user_data.get("rgb_image")
        if rgb_image is None:
            logger.warning(
                f"No RGB image in user_data for user {update.effective_user.id}"
            )
            await update.message.reply_text(t("rgb_not_found"))
            return WAITING_RGB

        await update.message.reply_text(t("processing"))

        # Convert NIR to grayscale if it's RGB
        nir_image = image.convert("L") if image.mode == "RGB" else image

        logger.info(
            f"Processing: RGB={rgb_image.size} mode={rgb_image.mode}, "
            f"NIR={nir_image.size} mode={nir_image.mode}"
        )

        # Preprocess: normalize to [0,1] and resize to 512×512
        rgbn_tensor = preprocess_image(rgb_image, nir_image)
        orig_size = rgb_image.size

        return await _run_analysis(update, context, rgbn_tensor, orig_size)

    except Exception as e:
        logger.error(f"Error during NIR processing: {e}", exc_info=True)
        await update.message.reply_text(t("error_analysis", error=str(e)))
        context.user_data.pop("rgb_image", None)
        return WAITING_RGB


async def skip_nir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Use grayscale of RGB as synthetic NIR."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    rgb_image = context.user_data.get("rgb_image")
    if rgb_image is None:
        await update.message.reply_text(t("skip_nir_no_rgb"))
        return WAITING_RGB

    await update.message.reply_text(t("skip_nir_warning"))

    nir_image = rgb_image.convert("L")
    rgbn_tensor = preprocess_image(rgb_image, nir_image)
    orig_size = rgb_image.size

    return await _run_analysis(update, context, rgbn_tensor, orig_size)


async def _run_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    rgbn_tensor: torch.Tensor,
    orig_size: tuple[int, int],
) -> int:
    """Run the full analysis pipeline and send results."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    lang = Locale.get_lang(update, context)

    try:
        # 1. NDVI
        ndvi_image = generate_ndvi_image(rgbn_tensor, lang=lang)
        ndvi = calculate_ndvi(rgbn_tensor)

        veg_healthy = (ndvi > 0.3).sum() / ndvi.size * 100
        veg_sparse = ((ndvi > 0.1) & (ndvi <= 0.3)).sum() / ndvi.size * 100
        non_veg = (ndvi <= 0.1).sum() / ndvi.size * 100

        ndvi_stats_text = (
            t("ndvi_stats_title")
            + t("ndvi_min", value=f"{ndvi.min():.3f}")
            + t("ndvi_max", value=f"{ndvi.max():.3f}")
            + t("ndvi_mean", value=f"{ndvi.mean():.3f}")
            + t("veg_health_title")
            + t("veg_healthy", value=f"{veg_healthy:.1f}")
            + t("veg_sparse", value=f"{veg_sparse:.1f}")
            + t("veg_nonveg", value=f"{non_veg:.1f}")
        )

        # 2. Prediction
        pred_mask = predict(model, rgbn_tensor, threshold=THRESHOLD)

        # 3. Overlay
        overlay_image = generate_prediction_overlay(rgbn_tensor, pred_mask)

        # 4. Detected classes
        detected_classes = []
        for i, name in enumerate(LABELS):
            pixels = int(pred_mask[i].sum().item())
            if pixels > 0:
                detected_classes.append((name, PALETTE[i], pixels))

        # 5. Legend
        legend_image = generate_legend_image(detected_classes, lang=lang)

        # 6. Detection text
        if not detected_classes:
            detection_text = t("no_anomalies")
        else:
            detection_text = t("detected_title")
            for name, color, pixels in detected_classes:
                translated_name = Locale.get_class_name(name, update, context)
                color_hex = "#{:02x}{:02x}{:02x}".format(*color)
                percentage = pixels / (IMAGE_WIDTH * IMAGE_HEIGHT) * 100
                emoji = _get_class_emoji(name)
                detection_text += t(
                    "detected_item",
                    emoji=emoji,
                    name=translated_name,
                    hex=color_hex,
                    rgb=str(color),
                    pixels=f"{pixels:,}",
                    percent=f"{percentage:.1f}",
                )

        # ============================================================
        # Send Results
        # ============================================================

        await update.message.reply_photo(
            photo=pil_to_bytes(ndvi_image),
            caption=t("ndvi_caption"),
        )
        await update.message.reply_text(ndvi_stats_text, parse_mode="Markdown")

        await update.message.reply_photo(
            photo=pil_to_bytes(overlay_image),
            caption=t("prediction_caption"),
        )

        await update.message.reply_text(detection_text, parse_mode="Markdown")

        # Summary
        summary = (
            t("analysis_complete")
            + t("input_size",
                orig_w=orig_size[0], orig_h=orig_size[1],
                target_w=IMAGE_WIDTH, target_h=IMAGE_HEIGHT)
            + t("threshold_info", threshold=THRESHOLD)
            + t("classes_detected", count=len(detected_classes), total=CLASSES_COUNT)
            + t("send_another")
        )
        await update.message.reply_text(summary, parse_mode="Markdown")

        # Clear image data but keep lang
        lang_pref = context.user_data.get("lang")
        context.user_data.clear()
        if lang_pref:
            context.user_data["lang"] = lang_pref

        return WAITING_RGB

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        await update.message.reply_text(t("error_analysis", error=str(e)))
        lang_pref = context.user_data.get("lang")
        context.user_data.clear()
        if lang_pref:
            context.user_data["lang"] = lang_pref
        return WAITING_RGB


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    text = (
        t("help_title")
        + t("help_commands")
        + t("help_workflow")
        + t("help_info", width=IMAGE_WIDTH, height=IMAGE_HEIGHT, classes=CLASSES_COUNT)
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def classes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    text = t("classes_title")
    for label in LABELS:
        color = PALETTE_DICT[label]
        emoji = _get_class_emoji(label)
        color_hex = "#{:02x}{:02x}{:02x}".format(*color)
        translated_name = Locale.get_class_name(label, update, context)
        text += f"  {emoji} `{translated_name:<24}` `{str(color):<18}` `{color_hex}`\n"
    await update.message.reply_text(text, parse_mode="Markdown")


async def threshold_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global THRESHOLD
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    args = context.args
    if args:
        try:
            new_threshold = float(args[0])
            if 0.0 < new_threshold < 1.0:
                THRESHOLD = new_threshold
                await update.message.reply_text(
                    t("threshold_updated", threshold=THRESHOLD),
                    parse_mode="Markdown",
                )
            else:
                await update.message.reply_text(t("threshold_error_range"))
        except ValueError:
            await update.message.reply_text(t("threshold_error_value"))
    else:
        await update.message.reply_text(
            t("threshold_current", threshold=THRESHOLD),
            parse_mode="Markdown",
        )


async def lang_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    await update.message.reply_text(t("lang_prompt"), parse_mode="Markdown")


async def lang_en(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["lang"] = "en"
    await update.message.reply_text(
        Locale.STRINGS["en"]["lang_switched"], parse_mode="Markdown"
    )


async def lang_ru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["lang"] = "ru"
    await update.message.reply_text(
        Locale.STRINGS["ru"]["lang_switched"], parse_mode="Markdown"
    )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    lang_pref = context.user_data.get("lang")
    context.user_data.clear()
    if lang_pref:
        context.user_data["lang"] = lang_pref
    await update.message.reply_text(t("cancelled"))
    return ConversationHandler.END


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error: {context.error}", exc_info=True)
    if update and update.message:
        try:
            t = lambda key, **kw: Locale.t(key, update, context, **kw)
            await update.message.reply_text(t("error_generic"))
        except Exception:
            pass


# ============================================================
# Main Entry Point
# ============================================================


def main():
    """Start the bot."""
    global model

    logger.info("Loading model...")
    model = load_model()
    logger.info(f"Model loaded on device: {DEVICE}")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # ============================================================
    # FIX: ConversationHandler restructured
    #
    # Problem: with allow_reentry=True and MessageHandler in entry_points,
    # sending the NIR image re-triggered the entry point (single_image_mode)
    # instead of the WAITING_NIR state handler (receive_nir).
    #
    # Solution:
    #   - Remove allow_reentry=True (default is False)
    #   - Images are NOT in entry_points — they're handled only in states
    #   - Entry is via /start or /analyze commands only
    #   - First image in WAITING_RGB: receive_rgb handles it
    #     (includes 4-channel auto-detection)
    #   - Second image in WAITING_NIR: receive_nir handles it
    #   - /start, /analyze in fallbacks allow restarting at any time
    # ============================================================

    IMAGE_FILTER = filters.PHOTO | filters.Document.IMAGE

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("analyze", analyze),
            # Allow sending image directly to start (enters WAITING_RGB handler)
            MessageHandler(IMAGE_FILTER, receive_rgb),
        ],
        states={
            WAITING_RGB: [
                MessageHandler(IMAGE_FILTER, receive_rgb),
            ],
            WAITING_NIR: [
                MessageHandler(IMAGE_FILTER, receive_nir),
                CommandHandler("skip_nir", skip_nir),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start", start),
            CommandHandler("analyze", analyze),
            CommandHandler("help", help_command),
            CommandHandler("skip_nir", skip_nir),
        ],
        # ⚠️ CRITICAL FIX: allow_reentry=False prevents the entry_point
        # MessageHandler from re-triggering when user sends NIR image
        allow_reentry=False,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("classes", classes_command))
    application.add_handler(CommandHandler("threshold", threshold_command))
    application.add_handler(CommandHandler("lang", lang_command))
    application.add_handler(CommandHandler("lang_en", lang_en))
    application.add_handler(CommandHandler("lang_ru", lang_ru))

    application.add_error_handler(error_handler)

    # Register bot commands via API
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setMyCommands"
        commands = [
            {"command": "start", "description": "Start / Начать"},
            {"command": "analyze", "description": "New analysis / Новый анализ"},
            {"command": "skip_nir", "description": "Skip NIR / Пропустить NIR"},
            {"command": "threshold", "description": "Set threshold / Порог"},
            {"command": "classes", "description": "Show classes / Классы"},
            {"command": "lang", "description": "Language / Язык"},
            {"command": "lang_en", "description": "English"},
            {"command": "lang_ru", "description": "Русский"},
            {"command": "help", "description": "Help / Помощь"},
            {"command": "cancel", "description": "Cancel / Отмена"},
        ]
        resp = requests.post(url, json={"commands": commands}, timeout=10)
        if resp.ok:
            logger.info("Bot commands registered.")
        else:
            logger.warning(f"Failed to register commands: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Exception setting commands: {e}")

    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
