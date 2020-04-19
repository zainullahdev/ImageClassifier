import argparse

def user_arguments():
    parser = argparse.ArgumentParser(description='This function takes a flower image and returns predictions.')
    parser.add_argument('image', type=str, 
                    help='Path of Image to predict')
    parser.add_argument('model_path', type=str, 
                    help='Path of model used for prediction in .h5 format')
    parser.add_argument('--top_k', type=int, const=1,nargs='?',help='Return Top K predictions, default 1')
    parser.add_argument('--category_names', type=str, const='label_map.json',nargs='?',help='Class Labels in .json format, default label_map.json if not specified')
    args = parser.parse_args()
    return args
