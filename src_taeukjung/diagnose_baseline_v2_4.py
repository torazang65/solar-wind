from diagnose_baseline_v2_2 import main
from model_baseline_v2_4 import SolarWindFixedLagMagnitudeTransformerV24


if __name__ == "__main__":
    main(
        model_class=SolarWindFixedLagMagnitudeTransformerV24,
        architecture_name="SolarWindFixedLagMagnitudeTransformerV24",
        file_stem="baseline_v2_4",
    )
