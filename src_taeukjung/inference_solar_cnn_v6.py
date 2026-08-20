from inference_solar_physics_v5 import main
from model_solar_cnn_v6 import SolarWindCNNTransformerV6


if __name__ == "__main__":
    main(
        model_class=SolarWindCNNTransformerV6,
        architecture_name="SolarWindCNNTransformerV6",
        file_stem="solar_cnn_v6",
    )
