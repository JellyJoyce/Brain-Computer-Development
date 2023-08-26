import os
import pickle
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters


def main():
    txt_files_folder_LRB = "./data/LRB_data"


    # loop to get all txt file
    txt_files_LRB = [f for f in os.listdir(txt_files_folder_LRB) if f.endswith(".txt")]

    for file in txt_files_LRB:
        with open(os.path.join(txt_files_folder_LRB, file), 'r') as f:
            lines = f.readlines()
        lines = [line.replace('\t', ' ') for line in lines]
        with open(os.path.join(txt_files_folder_LRB, file), 'w') as f:
            f.writelines(lines)

    # create a df to store extracted features
    features_df = pd.DataFrame()

    for txt_file in txt_files_LRB:
        print(txt_file)

        # txt -> df

        data = pd.read_csv(os.path.join(txt_files_folder_LRB, txt_file), sep=' ', header=None)
        
        data.columns = ['time', 'value']
        data['id'] = txt_file  # 为每个文件分配一个唯一的 ID
        
        # define the feature parameters to use
        fc_parameters = EfficientFCParameters()        
        
        extracted_features = extract_features(data, column_id='id', column_sort='time', default_fc_parameters=fc_parameters)

        # add label to df
        label = txt_file[0]
        extracted_features['label'] = label

        # add feature to features_df
        features_df = features_df.append(extracted_features)

    features_df.reset_index(drop=True, inplace=True)

    # Save extracted features to a pickle file
    with open('extracted_features.pkl', 'wb') as f:
        pickle.dump(features_df, f)
    
    print(features_df)

    


if __name__ == "__main__":
    main()