from inference_solar_physics_v5 import main
from model_baseline_v2_2 import SolarWindBaselineSpatialTransformerV22


if __name__ == "__main__":
    main(
        model_class=SolarWindBaselineSpatialTransformerV22,
        architecture_name="SolarWindBaselineSpatialTransformerV22",
        file_stem="baseline_v2_2",
        grid_label="spatial_grid",
    )
