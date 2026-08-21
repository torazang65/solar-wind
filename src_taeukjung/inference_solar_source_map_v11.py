from inference_solar_hybrid_v10 import main as run_inference
from model_solar_source_map_v11 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindSourceMapV11,
)


if __name__ == "__main__":
    run_inference(
        model_class=SolarWindSourceMapV11,
        architecture_name=ARCHITECTURE_NAME,
        file_stem=FILE_STEM,
    )
