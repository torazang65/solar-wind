from inference_solar_physics_v5 import main
from model_solar_arrival_v9 import SolarWindArrivalTCNV9


if __name__ == "__main__":
    main(
        model_class=SolarWindArrivalTCNV9,
        architecture_name="SolarWindArrivalTCNV9",
        file_stem="solar_arrival_v9",
    )
