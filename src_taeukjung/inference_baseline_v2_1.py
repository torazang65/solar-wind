from inference_solar_physics_v5 import main
from model_baseline_v2_1 import SolarWindBaselineTransformerV21


if __name__ == "__main__":
    main(
        model_class=SolarWindBaselineTransformerV21,
        architecture_name="SolarWindBaselineTransformerV21",
        file_stem="baseline_v2_1",
        grid_label="spatial_grid",
    )
