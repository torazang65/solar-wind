from diagnose_baseline_v2_2 import main
from model_baseline_v2_3 import SolarWindARNeuralTransformerV23


if __name__ == "__main__":
    main(
        model_class=SolarWindARNeuralTransformerV23,
        architecture_name="SolarWindARNeuralTransformerV23",
        file_stem="baseline_v2_3",
    )
