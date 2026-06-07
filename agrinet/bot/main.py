from dotenv import load_dotenv
import requests
import os
import io
import re
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from PIL import Image as PILImage, ImageFilter
import matplotlib

from telegram import Update
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
TIMEOUT = 30

IMAGE_WIDTH = IMAGE_HEIGHT = 512
RGBN_CHANNELS = 4

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

# Default backbone
DEFAULT_BACKBONE = "efficientnet-b2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Domain Adaptation Configuration
# ============================================================

AGV_MEAN = torch.tensor([0.3247, 0.3735, 0.3270, 0.4655], dtype=torch.float32)
AGV_STD = torch.tensor([0.1561, 0.1407, 0.1307, 0.1502], dtype=torch.float32)

KZ_MEAN = torch.tensor([0.3812, 0.4102, 0.2893, 0.5214], dtype=torch.float32)
KZ_STD = torch.tensor([0.1743, 0.1589, 0.1421, 0.1687], dtype=torch.float32)

REFERENCE_IMAGE_PATH = "reference_agv.pt"


class NormStrategy:
    SIMPLE = "simple"
    ZSCORE = "zscore"
    HISTOGRAM = "histogram"


DEFAULT_STRATEGY = NormStrategy.ZSCORE

# Conversation states
WAITING_RGB = 0
WAITING_NIR = 1

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# Model Manager — Multi-model support
# ============================================================


class ModelManager:
    """
    Manages multiple segmentation models loaded from disk.

    Scans for model files matching pattern: best_model_*_dataset.pth
    Extracts backbone name from filename and loads each model.
    """

    # Pattern: best_model_{backbone}_dataset.pth
    MODEL_FILE_PATTERN = "best_model_*_dataset.pth"
    MODEL_NAME_REGEX = re.compile(r"best_model_(.+?)_dataset\.pth$")

    def __init__(self, search_dir: str = ".", default_backbone: str = DEFAULT_BACKBONE):
        self._models: dict[str, nn.Module] = {}
        self._model_info: dict[str, dict] = {}
        self._active_model_name: str = ""
        self._default_backbone = default_backbone
        self._search_dir = search_dir

        self._discover_and_load_models()

    def _discover_and_load_models(self) -> None:
        """Scan directory for model files and load them all."""
        pattern = os.path.join(self._search_dir, self.MODEL_FILE_PATTERN)
        model_files = glob.glob(pattern)

        if not model_files:
            raise FileNotFoundError(
                f"No model files found matching '{pattern}'\n"
                f"Expected files like: best_model_efficientnet-b2_dataset.pth"
            )

        logger.info(f"Found {len(model_files)} model file(s) in '{self._search_dir}'")

        for filepath in sorted(model_files):
            filename = os.path.basename(filepath)
            match = self.MODEL_NAME_REGEX.match(filename)
            if not match:
                logger.warning(f"Skipping file (doesn't match pattern): {filename}")
                continue

            backbone_name = match.group(1)
            try:
                model, info = self._load_single_model(filepath, backbone_name)
                self._models[backbone_name] = model
                self._model_info[backbone_name] = info
                logger.info(
                    f"  ✓ Loaded: {backbone_name} "
                    f"(mIoU={info.get('best_miou', 'N/A'):.4f}, "
                    f"epoch={info.get('epoch', '?')})"
                )
            except Exception as e:
                logger.error(f"  ✗ Failed to load {backbone_name}: {e}")

        if not self._models:
            raise RuntimeError("No models could be loaded successfully.")

        # Set active model
        if self._default_backbone in self._models:
            self._active_model_name = self._default_backbone
        else:
            # Fallback to first available
            self._active_model_name = next(iter(self._models))
            logger.warning(
                f"Default backbone '{self._default_backbone}' not found. "
                f"Using '{self._active_model_name}' instead."
            )

        logger.info(f"Active model: {self._active_model_name}")

    def _load_single_model(
        self, filepath: str, backbone_name: str
    ) -> tuple[nn.Module, dict]:
        """Load a single model from checkpoint file."""
        model = smp.DeepLabV3Plus(
            encoder_name=backbone_name,
            encoder_weights=None,
            in_channels=RGBN_CHANNELS,
            classes=CLASSES_COUNT,
        ).to(DEVICE)

        checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        info = {
            "filepath": filepath,
            "backbone": backbone_name,
            "best_miou": checkpoint.get("best_miou", 0.0),
            "epoch": checkpoint.get("epoch", -1),
        }

        return model, info

    @property
    def active_model(self) -> nn.Module:
        """Get currently active model."""
        return self._models[self._active_model_name]

    @property
    def active_name(self) -> str:
        """Get name of currently active model."""
        return self._active_model_name

    @property
    def active_info(self) -> dict:
        """Get info about currently active model."""
        return self._model_info[self._active_model_name]

    @property
    def available_models(self) -> list[str]:
        """List all available model names."""
        return sorted(self._models.keys())

    def get_model_info(self, name: str) -> dict | None:
        """Get info for a specific model."""
        return self._model_info.get(name)

    def switch_model(self, backbone_name: str) -> bool:
        """
        Switch active model. Returns True if successful.
        Accepts partial matches (e.g., 'b2' matches 'efficientnet-b2').
        """
        # Exact match
        if backbone_name in self._models:
            self._active_model_name = backbone_name
            logger.info(f"Switched to model: {backbone_name}")
            return True

        # Partial match
        backbone_lower = backbone_name.lower()
        for name in self._models:
            if backbone_lower in name.lower():
                self._active_model_name = name
                logger.info(
                    f"Switched to model: {name} (matched from '{backbone_name}')"
                )
                return True

        return False

    def get_models_summary(self) -> str:
        """Get formatted summary of all models (ASCII only for logging)."""
        lines = []
        for name in self.available_models:
            info = self._model_info[name]
            active = " ← active" if name == self._active_model_name else ""
            lines.append(
                f"  {name}: mIoU={info['best_miou']:.4f}, "
                f"epoch={info['epoch']}{active}"
            )
        return "\n".join(lines)


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
                "using trained DeepLabV3+ neural networks.\n\n"
            ),
            "detectable_classes": "📋 *Detectable Classes:*\n",
            "how_to_use": (
                "\n📸 *How to use:*\n"
                "1️⃣ Send an RGB image (regular photo)\n"
                "2️⃣ Then send the corresponding NIR (Near-Infrared) image\n\n"
                "The bot will return:\n"
                "• NDVI visualization\n"
                "• Prediction overlay with detected anomalies\n"
                "• List of detected classes with pixel counts\n\n"
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
            "processing": "⏳ Processing with `{model}` ({strategy})... Please wait.",
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
            "prediction_caption": "🔍 Prediction Overlay — model: {model}",
            "no_anomalies": "✅ No anomalies detected at threshold `{threshold}`.",
            "detected_title": "🔍 *Detected Anomalies:*\n\n",
            "detected_item": (
                "  {emoji} *{name}*\n"
                "      🎨 Color: `{hex}`\n"
                "      📊 Pixels: `{pixels}` ({percent}%)\n\n"
            ),
            "legend_header": "\n📋 *Color Legend:*\n",
            "legend_item": "  {emoji} `{hex}` — {name}\n",
            "analysis_complete": "✅ *Analysis Complete!*\n\n",
            "input_size": "📐 Input: {orig_w}×{orig_h} → {target_w}×{target_h}\n",
            "threshold_info": "🎯 Threshold: `{threshold}`\n",
            "strategy_info": "🔧 Normalization: `{strategy}`\n",
            "model_info": "🧠 Model: `{model}` (mIoU: {miou})\n",
            "classes_detected": "🏷️ Detected: `{count}/{total}` classes\n\n",
            "send_another": "Send another image or /analyze to start over.",
            "help_title": "🛰️ *Agricultural Anomaly Detection Bot*\n\n",
            "help_commands": (
                "*Commands:*\n"
                "/start — Welcome message and instructions\n"
                "/analyze — Start new image analysis\n"
                "/skip\\_nir — Use grayscale as synthetic NIR\n"
                "/model — Show/switch model backbone\n"
                "/threshold — Show/set prediction threshold\n"
                "/strategy — Show/set normalization strategy\n"
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
                "*Architecture:* DeepLabV3+\n"
                "*Input size:* {width}×{height}\n"
                "*Classes:* {classes}\n"
                "*Models loaded:* {models}"
            ),
            "classes_title": "🏷️ *Detectable Anomaly Classes:*\n\n",
            "threshold_current": "🎯 Current threshold: `{threshold}`\n\nUsage: `/threshold 0.3` to change.",
            "threshold_updated": "✅ Threshold updated to `{threshold}`",
            "threshold_error_range": "❌ Threshold must be between 0 and 1.",
            "threshold_error_value": "❌ Invalid number. Usage: /threshold 0.5",
            "strategy_current": (
                "🔧 Current strategy: `{strategy}`\n\n"
                "*Available strategies:*\n"
                "  • `simple` — divide by 255 only (no adaptation)\n"
                "  • `zscore` — linear mean/std alignment (recommended for KZ)\n"
                "  • `histogram` — full distribution matching to training data\n\n"
                "Usage: `/strategy zscore`"
            ),
            "strategy_updated": "✅ Strategy updated to `{strategy}`",
            "strategy_error": "❌ Unknown strategy. Available: `simple`, `zscore`, `histogram`",
            "model_current": (
                "🧠 *Current model:* `{active}`\n"
                "   mIoU: `{miou:.4f}` | epoch: `{epoch}`\n\n"
                "*Available models:*\n{model_list}\n\n"
                "Usage: `/model {example}`"
            ),
            "model_switched": "✅ Model switched to `{model}` (mIoU: {miou:.4f})",
            "model_not_found": (
                "❌ Model `{name}` not found.\n\n"
                "*Available:*\n{model_list}\n\n"
                "💡 Partial match supported: `/model b3` → `efficientnet-b3`"
            ),
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
        },
        "ru": {
            "welcome_title": "🛰️ *Бот обнаружения аномалий в сельском хозяйстве*\n\n",
            "welcome_desc": (
                "Этот бот анализирует аэро/дрон-снимки для обнаружения аномалий "
                "сельскохозяйственных угодий с помощью обученных нейросетей DeepLabV3+.\n\n"
            ),
            "detectable_classes": "📋 *Обнаруживаемые классы:*\n",
            "how_to_use": (
                "\n📸 *Как использовать:*\n"
                "1️⃣ Отправьте RGB-снимок (обычное цветное фото)\n"
                "2️⃣ Затем отправьте соответствующий NIR-снимок (ближний инфракрасный)\n\n"
                "Бот вернёт:\n"
                "• Визуализацию NDVI\n"
                "• Наложение предсказания с обнаруженными аномалиями\n"
                "• Список обнаруженных классов с пикселями\n\n"
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
            "processing": "⏳ Обработка: `{model}` ({strategy})... Подождите.",
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
            "prediction_caption": "🔍 Предсказание — модель: {model}",
            "no_anomalies": "✅ Аномалий не обнаружено при пороге `{threshold}`.",
            "detected_title": "🔍 *Обнаруженные аномалии:*\n\n",
            "detected_item": (
                "  {emoji} *{name}*\n"
                "      🎨 Цвет: `{hex}`\n"
                "      📊 Пикселей: `{pixels}` ({percent}%)\n\n"
            ),
            "legend_header": "\n📋 *Цветовая легенда:*\n",
            "legend_item": "  {emoji} `{hex}` — {name}\n",
            "analysis_complete": "✅ *Анализ завершён!*\n\n",
            "input_size": "📐 Вход: {orig_w}×{orig_h} → {target_w}×{target_h}\n",
            "threshold_info": "🎯 Порог: `{threshold}`\n",
            "strategy_info": "🔧 Нормализация: `{strategy}`\n",
            "model_info": "🧠 Модель: `{model}` (mIoU: {miou})\n",
            "classes_detected": "🏷️ Обнаружено: `{count}/{total}` классов\n\n",
            "send_another": "Отправьте ещё изображение или /analyze для нового анализа.",
            "help_title": "🛰️ *Бот обнаружения аномалий в сельском хозяйстве*\n\n",
            "help_commands": (
                "*Команды:*\n"
                "/start — Приветствие и инструкции\n"
                "/analyze — Начать новый анализ\n"
                "/skip\\_nir — Использовать градации серого как NIR\n"
                "/model — Показать/сменить модель\n"
                "/threshold — Показать/установить порог\n"
                "/strategy — Показать/установить стратегию нормализации\n"
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
                "*Архитектура:* DeepLabV3+\n"
                "*Размер входа:* {width}×{height}\n"
                "*Классов:* {classes}\n"
                "*Загружено моделей:* {models}"
            ),
            "classes_title": "🏷️ *Обнаруживаемые классы аномалий:*\n\n",
            "threshold_current": "🎯 Текущий порог: `{threshold}`\n\nИспользование: `/threshold 0.3` для изменения.",
            "threshold_updated": "✅ Порог обновлён: `{threshold}`",
            "threshold_error_range": "❌ Порог должен быть от 0 до 1.",
            "threshold_error_value": "❌ Неверное число. Использование: /threshold 0.5",
            "strategy_current": (
                "🔧 Текущая стратегия: `{strategy}`\n\n"
                "*Доступные стратегии:*\n"
                "  • `simple` — только деление на 255 (без адаптации)\n"
                "  • `zscore` — линейное выравнивание mean/std (рекомендуется для KZ)\n"
                "  • `histogram` — полное совмещение распределений\n\n"
                "Использование: `/strategy zscore`"
            ),
            "strategy_updated": "✅ Стратегия обновлена: `{strategy}`",
            "strategy_error": "❌ Неизвестная стратегия. Доступные: `simple`, `zscore`, `histogram`",
            "model_current": (
                "🧠 *Текущая модель:* `{active}`\n"
                "   mIoU: `{miou:.4f}` | эпоха: `{epoch}`\n\n"
                "*Доступные модели:*\n{model_list}\n\n"
                "Использование: `/model {example}`"
            ),
            "model_switched": "✅ Модель переключена на `{model}` (mIoU: {miou:.4f})",
            "model_not_found": (
                "❌ Модель `{name}` не найдена.\n\n"
                "*Доступные:*\n{model_list}\n\n"
                "💡 Поддерживается частичное совпадение: `/model b3` → `efficientnet-b3`"
            ),
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
        return "ru"

    @classmethod
    def t(
        cls, key: str, update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs
    ) -> str:
        lang = cls.get_lang(update, context)
        strings = cls.STRINGS.get(lang, cls.STRINGS["ru"])
        template = strings.get(key, cls.STRINGS["ru"].get(key, f"[{key}]"))
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError):
                return template
        return template

    @classmethod
    def get_class_name(
        cls, label: str, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> str:
        lang = cls.get_lang(update, context)
        names = cls.CLASS_NAMES.get(lang, cls.CLASS_NAMES["ru"])
        return names.get(label, label)

    @classmethod
    def get_class_names_for_lang(cls, lang: str) -> dict:
        return cls.CLASS_NAMES.get(lang, cls.CLASS_NAMES["ru"])


# ============================================================
# Domain Adaptation Engine
# ============================================================


class DomainAdapter:
    """
    Applies domain adaptation to align KZ drone imagery
    with Agriculture-Vision (US) training distribution.
    """

    def __init__(
        self,
        src_mean: torch.Tensor = KZ_MEAN,
        src_std: torch.Tensor = KZ_STD,
        tgt_mean: torch.Tensor = AGV_MEAN,
        tgt_std: torch.Tensor = AGV_STD,
        reference_image: torch.Tensor | None = None,
    ):
        self.src_mean = src_mean
        self.src_std = src_std
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        self.reference_image = reference_image

    def adapt(self, img: torch.Tensor, strategy: str) -> torch.Tensor:
        """Apply domain adaptation strategy to a (4, H, W) tensor in [0, 1]."""
        if strategy == NormStrategy.SIMPLE:
            return img
        elif strategy == NormStrategy.ZSCORE:
            return self._apply_zscore(img)
        elif strategy == NormStrategy.HISTOGRAM:
            return self._apply_histogram(img)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to simple")
            return img

    def _apply_zscore(self, img: torch.Tensor) -> torch.Tensor:
        """Linear per-channel: (x - mu_kz) / sigma_kz * sigma_agv + mu_agv"""
        out = img.clone()
        for c in range(img.shape[0]):
            standardized = (img[c] - self.src_mean[c]) / (self.src_std[c] + 1e-8)
            out[c] = standardized * self.tgt_std[c] + self.tgt_mean[c]
        return out.clamp(0.0, 1.0)

    def _apply_histogram(self, img: torch.Tensor) -> torch.Tensor:
        """Non-parametric per-channel histogram matching."""
        try:
            from skimage.exposure import match_histograms
        except ImportError:
            logger.warning("scikit-image not installed, falling back to zscore")
            return self._apply_zscore(img)

        if self.reference_image is None:
            logger.warning(
                "No reference image for histogram matching, falling back to zscore"
            )
            return self._apply_zscore(img)

        src_np = img.permute(1, 2, 0).numpy()
        src_h, src_w = src_np.shape[:2]
        ref_resized = TF.resize(
            self.reference_image,
            [src_h, src_w],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        ref_np = ref_resized.permute(1, 2, 0).numpy()
        matched = match_histograms(src_np, ref_np, channel_axis=-1)
        return torch.from_numpy(matched.astype(np.float32)).permute(2, 0, 1).clamp(0, 1)


# ============================================================
# Global instances
# ============================================================

_reference_img = None
if os.path.exists(REFERENCE_IMAGE_PATH):
    try:
        _reference_img = torch.load(
            REFERENCE_IMAGE_PATH, map_location="cpu", weights_only=True
        )
        logger.info(f"Loaded reference image from {REFERENCE_IMAGE_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load reference image: {e}")

domain_adapter = DomainAdapter(reference_image=_reference_img)

# Will be initialized in main()
model_manager: ModelManager = None


# ============================================================
# Image Processing Utilities
# ============================================================


def _detect_scale_factor(
    arr: np.ndarray, pil_image: PILImage.Image | None = None
) -> float:
    """Detect 8-bit vs 16-bit and return divisor."""
    if pil_image is not None:
        mode = pil_image.mode
        dtype = np.array(pil_image).dtype
        if mode in ("I;16", "I;16B", "I;16L", "I") or dtype in (np.uint16, np.int32):
            return 65535.0
    if arr.max() > 255.0:
        return 65535.0
    return 255.0


def preprocess_image(
    rgb_image: PILImage.Image,
    nir_image: PILImage.Image,
    strategy: str = DEFAULT_STRATEGY,
) -> torch.Tensor:
    """
    Preprocess RGB + NIR with domain adaptation.
    Pipeline: detect bit depth → /scale → [0,1] → resize → adapt
    """
    rgb_arr = np.array(rgb_image.convert("RGB")).astype(np.float32)
    nir_arr = np.array(nir_image).astype(np.float32)

    rgb_scale = _detect_scale_factor(rgb_arr, rgb_image)
    nir_scale = _detect_scale_factor(nir_arr, nir_image)

    rgb_arr = rgb_arr / rgb_scale
    nir_arr = nir_arr / nir_scale

    if nir_arr.ndim == 3:
        nir_arr = nir_arr[..., 0]

    rgbn = np.concatenate([rgb_arr, nir_arr[..., np.newaxis]], axis=-1)
    img_tensor = torch.from_numpy(rgbn).permute(2, 0, 1).contiguous()

    img_tensor = TF.resize(
        img_tensor,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )

    img_tensor = domain_adapter.adapt(img_tensor, strategy)

    logger.info(
        f"Preprocessed: scale_rgb={rgb_scale:.0f}, scale_nir={nir_scale:.0f}, "
        f"strategy={strategy}, range=[{img_tensor.min():.3f}, {img_tensor.max():.3f}]"
    )
    return img_tensor


def preprocess_rgbn_4channel(
    rgbn_arr: np.ndarray,
    pil_image: PILImage.Image | None = None,
    strategy: str = DEFAULT_STRATEGY,
) -> torch.Tensor:
    """Preprocess 4-channel RGBN array with domain adaptation."""
    arr = rgbn_arr.astype(np.float32)
    scale = _detect_scale_factor(arr, pil_image)
    arr = arr / scale

    img_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    img_tensor = TF.resize(
        img_tensor,
        [IMAGE_HEIGHT, IMAGE_WIDTH],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )

    img_tensor = domain_adapter.adapt(img_tensor, strategy)
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


def generate_ndvi_image(rgbn_tensor: torch.Tensor) -> PILImage.Image:
    """Generate NDVI visualization (ASCII-only labels to avoid font issues)."""
    ndvi = calculate_ndvi(rgbn_tensor)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1.0, vmax=1.0)

    stats = f"min={ndvi.min():.3f}, max={ndvi.max():.3f}, mean={ndvi.mean():.3f}"
    ax.set_title(f"NDVI\n{stats}", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDVI")

    veg_healthy = (ndvi > 0.3).sum() / ndvi.size * 100
    veg_sparse = ((ndvi > 0.1) & (ndvi <= 0.3)).sum() / ndvi.size * 100
    non_veg = (ndvi <= 0.1).sum() / ndvi.size * 100

    stats_text = (
        f"Healthy (NDVI > 0.3): {veg_healthy:.1f}%\n"
        f"Sparse (0.1-0.3): {veg_sparse:.1f}%\n"
        f"Non-veg (<=0.1): {non_veg:.1f}%"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
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
def predict(rgbn_tensor: torch.Tensor, threshold: float = THRESHOLD) -> torch.Tensor:
    """Run prediction with the currently active model."""
    active = model_manager.active_model
    active.eval()
    img_batch = rgbn_tensor.unsqueeze(0).to(DEVICE)
    logits = active(img_batch)
    probs = torch.sigmoid(logits)
    return (probs > threshold)[0].cpu()


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
            color_overlay[mask_c] += np.array(PALETTE[c], dtype=np.float32)

    color_overlay = np.clip(color_overlay, 0, 255).astype(np.uint8)

    blended = rgb_uint8.copy()
    if mask_present.any():
        blended[mask_present] = (
            (1.0 - alpha) * rgb_uint8[mask_present].astype(np.float32)
            + alpha * color_overlay[mask_present].astype(np.float32)
        ).astype(np.uint8)

    return PILImage.fromarray(blended).filter(ImageFilter.GaussianBlur(radius=0.7))


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


def _color_to_hex(color: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _build_detection_text(
    detected_classes: list[tuple[str, tuple, int]],
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> str:
    """Build text-based detection report with legend (replaces legend image)."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    if not detected_classes:
        return t("no_anomalies", threshold=THRESHOLD)

    text = t("detected_title")

    for name, color, pixels in detected_classes:
        translated_name = Locale.get_class_name(name, update, context)
        color_hex = _color_to_hex(color)
        percentage = pixels / (IMAGE_WIDTH * IMAGE_HEIGHT) * 100
        emoji = _get_class_emoji(name)
        text += t(
            "detected_item",
            emoji=emoji,
            name=translated_name,
            hex=color_hex,
            rgb=str(color),
            pixels=f"{pixels:,}",
            percent=f"{percentage:.1f}",
        )

    # Text legend
    text += t("legend_header")
    for name, color, _ in detected_classes:
        translated_name = Locale.get_class_name(name, update, context)
        emoji = _get_class_emoji(name)
        color_hex = _color_to_hex(color)
        text += t("legend_item", emoji=emoji, hex=color_hex, name=translated_name)

    return text


def _build_model_list(active_name: str) -> str:
    """Build formatted model list for display."""
    lines = []
    for name in model_manager.available_models:
        info = model_manager.get_model_info(name)
        marker = " ← ✓" if name == active_name else ""
        lines.append(f"  • `{name}` — mIoU: `{info['best_miou']:.4f}`{marker}")
    return "\n".join(lines)


def pil_to_bytes(img: PILImage.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Helper: Download image from Telegram message
# ============================================================


async def _download_image_from_message(update: Update) -> PILImage.Image | None:
    if (
        update.message.document
        and update.message.document.mime_type
        and update.message.document.mime_type.startswith("image/")
    ):
        file = await update.message.document.get_file()
    elif update.message.photo:
        file = await update.message.photo[-1].get_file()
    else:
        return None
    file_bytes = await file.download_as_bytearray()
    return PILImage.open(io.BytesIO(file_bytes))


def _get_user_strategy(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("strategy", DEFAULT_STRATEGY)


def _get_user_model(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Get user's preferred model. Falls back to manager's active model."""
    return context.user_data.get("model", model_manager.active_name)


def _activate_user_model(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ensure model_manager is set to user's preference for this request."""
    preferred = _get_user_model(context)
    if preferred != model_manager.active_name:
        if not model_manager.switch_model(preferred):
            # Reset user preference if model no longer exists
            context.user_data["model"] = model_manager.active_name


# ============================================================
# Telegram Bot Handlers
# ============================================================


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    text = t("welcome_title")
    text += t("welcome_desc")
    text += t("detectable_classes")

    for label in LABELS:
        color = PALETTE_DICT[label]
        emoji = _get_class_emoji(label)
        color_hex = _color_to_hex(color)
        class_name = Locale.get_class_name(label, update, context)
        text += f"  {emoji} {class_name} — `{color_hex}`\n"

    text += t("how_to_use")

    await update.message.reply_text(text, parse_mode="Markdown")
    return WAITING_RGB


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang = context.user_data.get("lang")
    strategy = context.user_data.get("strategy", DEFAULT_STRATEGY)
    user_model = context.user_data.get("model", model_manager.active_name)
    context.user_data.clear()
    if lang:
        context.user_data["lang"] = lang
    context.user_data["strategy"] = strategy
    context.user_data["model"] = user_model

    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    await update.message.reply_text(t("send_rgb"), parse_mode="Markdown")
    return WAITING_RGB


async def receive_rgb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    try:
        image = await _download_image_from_message(update)
        if image is None:
            await update.message.reply_text(t("send_image_error"))
            return WAITING_RGB

        img_arr = np.array(image)
        strategy = _get_user_strategy(context)

        # 4-channel PNG
        if img_arr.ndim == 3 and img_arr.shape[2] == 4:
            await update.message.reply_text(t("detected_4ch"))
            rgbn_tensor = preprocess_rgbn_4channel(img_arr, image, strategy)
            orig_size = (img_arr.shape[1], img_arr.shape[0])
            return await _run_analysis(update, context, rgbn_tensor, orig_size)

        # Store RGB, wait for NIR
        rgb_image = image.convert("RGB")
        context.user_data["rgb_image"] = rgb_image

        logger.info(
            f"RGB stored: user={update.effective_user.id}, "
            f"size={rgb_image.size}, model={_get_user_model(context)}"
        )

        await update.message.reply_text(
            t("send_nir", width=rgb_image.size[0], height=rgb_image.size[1]),
            parse_mode="Markdown",
        )
        return WAITING_NIR

    except Exception as e:
        logger.error(f"Error receiving RGB: {e}", exc_info=True)
        await update.message.reply_text(t("error_processing", error=str(e)))
        return WAITING_RGB


async def receive_nir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    try:
        image = await _download_image_from_message(update)
        if image is None:
            await update.message.reply_text(t("send_image_error"))
            return WAITING_NIR

        rgb_image = context.user_data.get("rgb_image")
        if rgb_image is None:
            await update.message.reply_text(t("rgb_not_found"))
            return WAITING_RGB

        strategy = _get_user_strategy(context)
        user_model = _get_user_model(context)
        await update.message.reply_text(
            t("processing", model=user_model, strategy=strategy)
        )

        nir_image = image.convert("L") if image.mode == "RGB" else image

        rgbn_tensor = preprocess_image(rgb_image, nir_image, strategy)
        orig_size = rgb_image.size

        return await _run_analysis(update, context, rgbn_tensor, orig_size)

    except Exception as e:
        logger.error(f"Error during NIR processing: {e}", exc_info=True)
        await update.message.reply_text(t("error_analysis", error=str(e)))
        context.user_data.pop("rgb_image", None)
        return WAITING_RGB


async def skip_nir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    rgb_image = context.user_data.get("rgb_image")
    if rgb_image is None:
        await update.message.reply_text(t("skip_nir_no_rgb"))
        return WAITING_RGB

    await update.message.reply_text(t("skip_nir_warning"))

    strategy = _get_user_strategy(context)
    nir_image = rgb_image.convert("L")
    rgbn_tensor = preprocess_image(rgb_image, nir_image, strategy)
    orig_size = rgb_image.size

    return await _run_analysis(update, context, rgbn_tensor, orig_size)


async def _run_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    rgbn_tensor: torch.Tensor,
    orig_size: tuple[int, int],
) -> int:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    strategy = _get_user_strategy(context)

    # Switch to user's preferred model
    _activate_user_model(context)
    active_name = model_manager.active_name
    active_info = model_manager.active_info

    try:
        # 1. NDVI
        ndvi_image = generate_ndvi_image(rgbn_tensor)
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
        pred_mask = predict(rgbn_tensor, threshold=THRESHOLD)

        # 3. Overlay
        overlay_image = generate_prediction_overlay(rgbn_tensor, pred_mask)

        # 4. Detected classes
        detected_classes = []
        for i, name in enumerate(LABELS):
            pixels = int(pred_mask[i].sum().item())
            if pixels > 0:
                detected_classes.append((name, PALETTE[i], pixels))

        # 5. Detection text (replaces legend image)
        detection_text = _build_detection_text(detected_classes, update, context)

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
            caption=t("prediction_caption", model=active_name),
        )

        await update.message.reply_text(detection_text, parse_mode="Markdown")

        # Summary
        miou_str = f"{active_info['best_miou']:.4f}"
        summary = (
            t("analysis_complete")
            + t(
                "input_size",
                orig_w=orig_size[0],
                orig_h=orig_size[1],
                target_w=IMAGE_WIDTH,
                target_h=IMAGE_HEIGHT,
            )
            + t("model_info", model=active_name, miou=miou_str)
            + t("threshold_info", threshold=THRESHOLD)
            + t("strategy_info", strategy=strategy)
            + t("classes_detected", count=len(detected_classes), total=CLASSES_COUNT)
            + t("send_another")
        )
        await update.message.reply_text(summary, parse_mode="Markdown")

        # Preserve preferences
        _preserve_preferences(context)
        return WAITING_RGB

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        await update.message.reply_text(t("error_analysis", error=str(e)))
        _preserve_preferences(context)
        return WAITING_RGB


def _preserve_preferences(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear temp data but keep user preferences."""
    lang_pref = context.user_data.get("lang")
    strat_pref = context.user_data.get("strategy", DEFAULT_STRATEGY)
    model_pref = context.user_data.get("model", model_manager.active_name)
    context.user_data.clear()
    if lang_pref:
        context.user_data["lang"] = lang_pref
    context.user_data["strategy"] = strat_pref
    context.user_data["model"] = model_pref


# ============================================================
# Command Handlers
# ============================================================


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    text = (
        t("help_title")
        + t("help_commands")
        + t("help_workflow")
        + t(
            "help_info",
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            classes=CLASSES_COUNT,
            models=len(model_manager.available_models),
        )
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def classes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    text = t("classes_title")
    for label in LABELS:
        color = PALETTE_DICT[label]
        emoji = _get_class_emoji(label)
        color_hex = _color_to_hex(color)
        translated_name = Locale.get_class_name(label, update, context)
        text += f"  {emoji} `{color_hex}` — {translated_name}\n"
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


async def strategy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    t = lambda key, **kw: Locale.t(key, update, context, **kw)
    current = _get_user_strategy(context)

    args = context.args
    if args:
        new_strategy = args[0].lower().strip()
        valid = {NormStrategy.SIMPLE, NormStrategy.ZSCORE, NormStrategy.HISTOGRAM}
        if new_strategy in valid:
            context.user_data["strategy"] = new_strategy
            await update.message.reply_text(
                t("strategy_updated", strategy=new_strategy),
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(t("strategy_error"), parse_mode="Markdown")
    else:
        await update.message.reply_text(
            t("strategy_current", strategy=current),
            parse_mode="Markdown",
        )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model command — list available models or switch."""
    t = lambda key, **kw: Locale.t(key, update, context, **kw)

    current = _get_user_model(context)
    current_info = model_manager.get_model_info(current) or model_manager.active_info

    args = context.args
    if args:
        requested = args[0].strip()

        # Try to find a match
        # First: exact match in available models
        found = None
        for name in model_manager.available_models:
            if requested.lower() == name.lower():
                found = name
                break

        # Second: partial match
        if not found:
            requested_lower = requested.lower()
            for name in model_manager.available_models:
                if requested_lower in name.lower():
                    found = name
                    break

        if found:
            context.user_data["model"] = found
            info = model_manager.get_model_info(found)
            await update.message.reply_text(
                t("model_switched", model=found, miou=info["best_miou"]),
                parse_mode="Markdown",
            )
        else:
            model_list = _build_model_list(current)
            await update.message.reply_text(
                t("model_not_found", name=requested, model_list=model_list),
                parse_mode="Markdown",
            )
    else:
        model_list = _build_model_list(current)
        example = (
            model_manager.available_models[0]
            if model_manager.available_models
            else "backbone"
        )
        await update.message.reply_text(
            t(
                "model_current",
                active=current,
                miou=current_info["best_miou"],
                epoch=current_info["epoch"],
                model_list=model_list,
                example=example,
            ),
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
    _preserve_preferences(context)
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
    global model_manager

    # ─── Load ALL models from current directory ───
    logger.info("=" * 60)
    logger.info("Discovering and loading models...")
    logger.info("=" * 60)

    model_manager = ModelManager(
        search_dir=".",
        default_backbone=DEFAULT_BACKBONE,
    )

    logger.info(f"\nLoaded models summary:")
    logger.info(f"\n{model_manager.get_models_summary()}")
    logger.info(f"\nActive model: {model_manager.active_name}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Default strategy: {DEFAULT_STRATEGY}")
    logger.info("=" * 60)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    IMAGE_FILTER = filters.PHOTO | filters.Document.IMAGE

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("analyze", analyze),
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
        allow_reentry=False,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("classes", classes_command))
    application.add_handler(CommandHandler("threshold", threshold_command))
    application.add_handler(CommandHandler("strategy", strategy_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("lang", lang_command))
    application.add_handler(CommandHandler("lang_en", lang_en))
    application.add_handler(CommandHandler("lang_ru", lang_ru))

    application.add_error_handler(error_handler)

    # Register bot commands
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setMyCommands"
        commands = [
            {"command": "start", "description": "Start / Начать"},
            {"command": "analyze", "description": "New analysis / Новый анализ"},
            {"command": "model", "description": "Switch model / Сменить модель"},
            {"command": "skip_nir", "description": "Skip NIR / Пропустить NIR"},
            {"command": "threshold", "description": "Set threshold / Порог"},
            {"command": "strategy", "description": "Normalization / Нормализация"},
            {"command": "classes", "description": "Show classes / Классы"},
            {"command": "lang", "description": "Language / Язык"},
            {"command": "lang_en", "description": "English"},
            {"command": "lang_ru", "description": "Русский"},
            {"command": "help", "description": "Help / Помощь"},
            {"command": "cancel", "description": "Cancel / Отмена"},
        ]
        resp = requests.post(url, json={"commands": commands}, timeout=TIMEOUT)
        if resp.ok:
            logger.info("Bot commands registered.")
        else:
            logger.warning(f"Failed to register commands: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Exception setting commands: {e}")

    logger.info("Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, timeout=TIMEOUT)


if __name__ == "__main__":
    main()
