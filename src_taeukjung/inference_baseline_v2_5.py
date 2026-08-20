from inference_solar_physics_v5 import main
from model_baseline_v2_5 import SolarWindDeepFixedLagTransformerV25


if __name__ == "__main__":
    main(
        model_class=SolarWindDeepFixedLagTransformerV25,
        architecture_name="SolarWindDeepFixedLagTransformerV25",
        file_stem="baseline_v2_5",
        grid_label="spatial_grid",
    )
