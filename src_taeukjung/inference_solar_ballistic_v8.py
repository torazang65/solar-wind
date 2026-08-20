from inference_solar_physics_v5 import main
from model_solar_ballistic_v8 import SolarWindBallisticTransformerV8


if __name__ == "__main__":
    main(
        model_class=SolarWindBallisticTransformerV8,
        architecture_name="SolarWindBallisticTransformerV8",
        file_stem="solar_ballistic_v8",
    )
