from pathlib import Path
import yaml

_yaml_path = Path(__file__).with_name("coefficients.yaml")

with _yaml_path.open("r", encoding="utf-8") as f:
    elements_info = yaml.safe_load(f)