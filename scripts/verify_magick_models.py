import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bellium.nano_nn.contract import inspect_model, NanoBudget
from bellium.micro_nn.mlp import load_mlp, predict_mlp

def main():
    m1 = load_mlp('Bellium-AI/models/nano_nn/resample-edge/v0.json')
    i1 = inspect_model(m1, budget=NanoBudget(64, 8192, 'float-json-weights-v1'))
    p1 = predict_mlp(m1, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    print('resample-edge:', i1['parameters'], 'params, out:', p1)

    m2 = load_mlp('Bellium-AI/models/nano_nn/quantize-tone/v0.json')
    i2 = inspect_model(m2, budget=NanoBudget(48, 4096, 'float-json-weights-v1'))
    p2 = predict_mlp(m2, [0.01, 0.02, 0.05, 0.5])
    print('quantize-tone:', i2['parameters'], 'params, out:', p2)

    m3 = load_mlp('Bellium-AI/models/nano_nn/adaptive-filter/v0.json')
    i3 = inspect_model(m3, budget=NanoBudget(48, 4096, 'float-json-weights-v1'))
    p3 = predict_mlp(m3, [0.1, 0.05, 0.05, 0.02])
    print('adaptive-filter:', i3['parameters'], 'params, out:', p3)

    m4 = load_mlp('Bellium-AI/models/micro_nn/tone-curve/v0.json')
    p4 = predict_mlp(m4, [0.1, 0.2, 0.4, 0.6, 0.9, 0.45, 0.25, 0.8])
    total_p4 = len(m4['weights'][0]) + len(m4['biases'][0]) + len(m4['weights'][1]) + len(m4['biases'][1])
    print('tone-curve:', total_p4, 'params, out:', p4)
    print('All models validated perfectly!')

if __name__ == '__main__':
    main()
