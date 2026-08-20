from inference_solar_physics_v5 import main
from model_solar_factorized_v7 import SolarWindFactorizedTransformerV7


if __name__ == "__main__":
    main(
        model_class=SolarWindFactorizedTransformerV7,
        architecture_name="SolarWindFactorizedTransformerV7",
        file_stem="solar_factorized_v7",
    )
