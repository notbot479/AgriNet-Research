import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Dict, Any, List
import os



def normalize_key(key: str) -> str:
    """Convert metric names to snake_case."""
    key = key.strip().lower()
    key = re.sub(r'[^0-9a-z]+', '_', key)
    return key.strip('_') or 'unnamed'


def coerce_value(value: str) -> Any:
    """Try to cast to int/float, otherwise return the original string."""
    value = value.strip()
    if re.fullmatch(r'-?\d+', value):
        return int(value)
    if re.fullmatch(r'-?\d*\.\d+(e[+-]?\d+)?', value, flags=re.IGNORECASE):
        return float(value)
    return value


def parse_chunk(chunk: str, record: Dict[str, Any]) -> None:
    """Parse a single chunk (already stripped)."""
    if ':' in chunk:
        key, value = chunk.split(':', 1)
        record[normalize_key(key)] = coerce_value(value)
        return

    # Special handling for things like "Epoch [5/100]"
    m = re.match(r'([^\s\[]+)\s*\[(\d+)\s*/\s*(\d+)\]', chunk)
    if m:
        name = normalize_key(m.group(1))
        record[f'{name}_current'] = int(m.group(2))
        record[f'{name}_total'] = int(m.group(3))
        return

    # Fallback: keep the raw chunk so we don’t lose information
    record.setdefault('raw_chunks', []).append(chunk)


def parse_line(line: str) -> Dict[str, Any]:
    """Parse a full log line into a dict."""
    record: Dict[str, Any] = {}
    chunks = [chunk.strip() for chunk in line.split('|')]
    for chunk in chunks:
        if chunk:
            parse_chunk(chunk, record)

    if 'raw_chunks' in record:
        record['raw_chunks'] = ' | '.join(record['raw_chunks'])
    return record


def parse_log(log_text: str) -> pd.DataFrame:
    """Turn the entire log into a tidy DataFrame."""
    rows: List[Dict[str, Any]] = []

    for line in log_text.splitlines():
        line = line.strip()
        if not line or line.startswith('✓'):  # skip empty lines / “saved best model” lines
            continue

        record = parse_line(line)
        if record:
            rows.append(record)

    df = pd.DataFrame(rows)
    # Optional: drop duplicate epochs if they exist
    if 'epoch_current' in df.columns:
        df = df.drop_duplicates(subset='epoch_current', keep='first')

    return df.reset_index(drop=True)


# --------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)

with open(os.path.join(CURRENT_DIR, 'training_logs.txt'), 'r') as f:
    raw_log = f.read()

df = parse_log(raw_log)
print(df.head())

# --- tidy data for plotting ---
loss_df = (
    df.melt(
        id_vars='epoch_current',
        value_vars=['train_loss', 'val_loss'],
        var_name='split',
        value_name='loss'
    )
)

# --- plot ---
sns.set_theme(style='whitegrid')
plt.figure(figsize=(7, 4))
sns.lineplot(
    data=loss_df,
    x='epoch_current',
    y='loss',
    hue='split',
    marker='o'
)
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()