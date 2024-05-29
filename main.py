import json

import numpy as np

from src import star, dash, utils

def main(dataset_path):
    xyDim = 112
    strides = 2
    #Load model info, example: tless/models_eval/models_info.json
    with open(f'{dataset_path}/models_eval/models_info.json') as f:
        jsondata = json.load(f)
    for key, model_info in jsondata.items():
        try:
            model_info["symmetries_discrete"] = np.array([
                np.array(sym).reshape((4, 4)) for sym in model_info["symmetries_discrete"]
            ])
            has_discrete_sym = True
        except KeyError:
            #model has no discrete symmetries
            has_discrete_sym = False
        # isvalid, depth, segmentation are used for loss calculation, but not used for ground truth
        inputs, po_image, isvalid, depth, segmentation = utils.dataset_conversion_layers(xyDim, xyDim, model_info, strides)
        if has_discrete_sym:
            offset = model_info["symmetries_discrete"][0][:3,-1] / 2.0
        else:
            offset = 0
        valid_dash = dash.calculate(
            inputs['rotationmatrix'],
            po_image,
            offset = offset
        )
        valid_star = star.calculate(model_info[key], po_image)
        #ToDo: Save star and dash images


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-dpath", "--dataset-path", type=str, default=".", help="The path to the used dataset")
    args = parser.parse_args()
    main(
        dataset_path=args.dataset_path
    )