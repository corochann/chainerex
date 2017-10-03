"""XGBoost util scripts"""
import numpy
import pandas
from chainerex.utils.log import save_json



def convert_evals_result_to_log_report(filepath, evals_result):
    """Convert XGBoost's evals_result to Chainer's LogReport format
    
    Convert XGBoost's evals_result to Chainer's LogReport format 
    and save it to `filepath`
    
    Args:
        filepath (str): 
        evals_result (dict): 

    Returns:

    """
    # --- Preprocess evals_result to key name concatenated dict format ---
    preprocessed_dict = {}
    for item in evals_result.items():
        key, value = item
        if isinstance(value, dict):
            for key2, value2 in value.items():
                # print('key, key2, value2', key, key2, value2)
                target_value = value2
                target_key = key + '/' + key2
                preprocessed_dict.update({target_key: target_value})
        else:
            print('[Error] Unhandled case1, value is of type', type(value))
    df = pandas.DataFrame(preprocessed_dict)
    df['iteration'] = numpy.arange(df.shape[0]) + 1
    df['epoch'] = numpy.arange(df.shape[0]) + 1

    log_report_list = []
    for key, row in df.iterrows():
        log_report_list.append(row.to_dict())
    save_json(filepath, log_report_list)


if __name__ == '__main__':
    # XGBoost format
    evals_result = {
        "training": {
            "mae": [
                0.446483,
                0.400076
            ],
            "mae2": [
                1.446483,
                1.400076
            ],
        },
        "valid_0": {
            "mae": [
                0.440902,
                0.394548
            ],
            "mae2": [
                10.440902,
                10.394548
            ]
        }
    }
    # LogReport format should be...
    # log_report_result = {
    #     {
    #         "training/mae": 0.446483,
    #         "valid_0/mae": 0.440902,
    #         "epoch": 1,
    #         "iteration": 1,
    #         "elapsed_time": 0,
    #     },
    #     {
    #         "training/mae": 0.400076,
    #         "valid_0/mae": 0.394548,
    #         "epoch": 2,
    #         "iteration": 2,
    #         "elapsed_time": 0,
    #     }
    # }
    convert_evals_result_to_log_report('log', evals_result)
