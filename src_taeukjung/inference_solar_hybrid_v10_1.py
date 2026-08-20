from inference_solar_hybrid_v10 import main
from model_solar_hybrid_v10_1 import (
    ARCHITECTURE_NAME,
    FILE_STEM,
    SolarWindSelectiveHybridV101,
)


if __name__ == "__main__":
    main(
        model_class=SolarWindSelectiveHybridV101,
        architecture_name=ARCHITECTURE_NAME,
        file_stem=FILE_STEM,
    )
