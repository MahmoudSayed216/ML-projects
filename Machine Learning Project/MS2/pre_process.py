import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder
import pickle as pkl

popular_degrees = ['MBBS', 'FCPS', 'MCPS', 'MS', 'MD', 'FRCS']



def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    df.ffill(inplace=True)
    return df




def addr_to_hospital_name(df: pd.DataFrame) ->pd.DataFrame:
    first_split = df['Hospital Address'].str.split(',', expand=True)
    second_split = first_split[0].str.split(';', expand=True)
    new_column_zero = second_split.iloc[:, 0]
    df_concatenated = pd.concat([df, new_column_zero], axis=1)
    df = df_concatenated.rename(columns={0: 'Hospital Name'})
    df.drop('Hospital Address', axis=1, inplace=True)
    return df




def encode_link(link):
    if link != 'No Link Available':
        return 1
    else:
        return 0




def encode_qualification(df: pd.DataFrame) -> pd.DataFrame:

    qualifications_df = pd.DataFrame({'MBBS':[], 'FCPS':[], 'MCPS':[], 'MS':  [], 'MD':  [], 'FRCS':[]})

    for  index in range(df.shape[0]):
        current_qualifications = df.iloc[index]['Doctor Qualification']
        row = []
        for qualification in popular_degrees:
            if current_qualifications.find(qualification) != -1:
                row.append(1)
            else:
                row.append(0)
        qualifications_df.loc[len(qualifications_df)] = row 
    

    df[popular_degrees] = qualifications_df[popular_degrees].copy()
    df.drop(['Doctor Qualification'], axis=1, inplace=True)

    return df






def encode_spec(df: pd.DataFrame) -> pd.DataFrame:
    specializations = []
    for cell in df['Specialization']:
        specs = cell.split(', ')
        for spec in specs:
            specializations.append(spec)

    unique_specializations = set(specializations)
    specs_cols = {spec : [] for spec in unique_specializations }
    specs_df = pd.DataFrame(specs_cols)

    for  index in range(df.shape[0]):
        current_spec = df.iloc[index]['Specialization']
        row = []
        for specialization in unique_specializations:
            if current_spec.find(specialization) != -1:
                row.append(1)
            else:
                row.append(0)
        specs_df.loc[len(specs_df)] = row 
        
    specs_df.head()
    df[list(unique_specializations)] = specs_df[list(unique_specializations)]
    
    
    return df





def pre_process(df: pd.DataFrame, label: str) -> pd.DataFrame:
    
    mean_encoder_c, mean_encoder_s, mean_encoder_h = None, None, None
    RS = None

    if label == 'Fee Category':
        mean_encoder_c = pkl.load(open("mean_encoder_c_classif.pkl", "rb"))
        mean_encoder_s = pkl.load(open("mean_encoder_s_classif.pkl", "rb"))
        mean_encoder_h = pkl.load(open("mean_encoder_h_classif.pkl", "rb"))
    elif label == 'Fee(PKR)':
        mean_encoder_c = pkl.load(open("mean_encoder_c_reg.pkl", "rb"))
        mean_encoder_s = pkl.load(open("mean_encoder_s_reg.pkl", "rb"))
        mean_encoder_h = pkl.load(open("mean_encoder_h_reg.pkl", "rb"))

    df = handle_null_values(df)

    df=df.drop(['Doctor Name'],axis=1)
    
    try:
        df=df.drop(['index'],axis=1)
    except:
        pass

    df = addr_to_hospital_name(df)

    df['Doctors Link'] = df['Doctors Link'].apply(lambda x: encode_link(x))

    df = encode_qualification(df)

    df.rename(columns={'Experience(Years)':'EXP(YRs)',
            'Total_Reviews' : '#Reviews',
            f'Patient Satisfaction Rate(%age)': 'Satisfaction Rate'}, inplace=True)

    if label == 'Fee Category':
        df['Fee Category'].replace('Cheap', 0, inplace=True)
        df['Fee Category'].replace('Medium-Priced', 1, inplace=True)
        df['Fee Category'].replace('Expensive', 2, inplace=True)


    df['City'] = mean_encoder_c.transform(df['City'])
    df['Specialization'] = mean_encoder_s.transform(df['Specialization'])


    indices = []
    for i in range(df.shape[0]):
        if df['Hospital Name'].iloc[i] == 'No Address Available':
            indices.append(i)

    df['Hospital Name'] = mean_encoder_h.transform(df['Hospital Name'])


    for index in indices:
        df['Hospital Name'].iloc[index] = 0
            
    try:
        df=df.drop(['index'],axis=1)
    except:
        pass
    final_features = []
    if label == 'Fee(PKR)':
        final_features=  ['MBBS',	'FCPS',	'MCPS', 	'MS',	'MD',	'FRCS',	'City',	'Specialization',	'EXP(YRs)',	'Hospital Name',	'#Reviews',	'Doctors Link',	'Fee(PKR)']
    elif label == 'Fee Category':
        final_features = ['City',	'Specialization',	'EXP(YRs)',	'Hospital Name',	'#Reviews',	'Doctors Link',	'MBBS',	'FCPS',	'MCPS',	'MS',	'MD',	'FRCS',	'Fee Category']
                         

    final_df = df[final_features] 
    # final_df[popular_degrees] = df[popular_degrees]
    return final_df
