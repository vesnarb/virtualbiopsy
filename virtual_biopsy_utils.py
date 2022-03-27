
import pickle as pkl
from datetime import datetime
import numpy as np
import pandas as pd
import os


def load_sentara(path='../pkls/sentara.pkl', overwrite=False): 
    """ 
    Returns sentara dataframe from existing pickle or from processed file (and dumps to pickle for next time).
    Args: path (str): path to existing/desired sentara data pickle 
    overwrite (boolean): whether to overwrite file if exists 
    Returns (pd.DataFrame): sentara data indexed according to patient ID and index date 
    """

    if os.path.isfile(path) and not overwrite: 
        sen_data = pkl.load(open(path, 'rb')) 
    else: 
        sen_data = pd.read_csv('../input_files/fx_sentara_cohort_processed.csv', 
                                 parse_dates = ['study_date'],
                                 index_col = [0, 'patient_id', 'study_date'])
        pkl.dump(sen_data, open(path, 'wb')) 
    print('shape:' , sen_data.shape) 
    return sen_data


def load_maccabi(path='../pkls/maccabi_processed.pkl', overwrite=False): 
    """ 
    Returns maccabi dataframe from existing pickle.
    Args: path (str): path to existing/desired sentara data pickle 
    overwrite (boolean): whether to overwrite file if exists 
    Returns (pd.DataFrame): maccabi data indexed according to patient ID and index date 
    """

    if os.path.isfile(path) and not overwrite: 
        mac_data = pkl.load(open(path, 'rb')) 
    else: 
        mac_data = pd.read_csv('../input_files/maccabi_processed.csv', 
                                 parse_dates = ['study_date'],
                                 index_col = ['patient_id', 'study_date'])
    print('shape:' , mac_data.shape) 
    return mac_data

def split_sentara(sen_data: pd.DataFrame, train_path = '../pkls/sentara_train.pkl', val_path = '../pkls/sentara_val.pkl', 
                  test_path = '../pkls/sentara_test.pkl', overwrite = False):
    """
    Splits Sentara dataframe into train, validation and test sets.
    If pickles already exist and there's no need to overwrite, load directly from pickles.
    Args:
    sen_data: sentara dataframe to split.
    train_path (str): path to train set
    val_path (str): path to validation set
    test_path (str): path to test set
    overwrite (boolea): whether to overwrite existing files.
    Returns (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series): matrix (x) and labels (y) for train, 
    vat, test, respectively 
    """
    
    if os.path.isfile(train_path) and not overwrite: 
        x_train, y_train = pkl.load(open(train_path, 'rb')) 
        x_val, y_val = pkl.load(open(val_path, 'rb')) 
        x_test, y_test = pkl.load(open(test_path, 'rb')) 
    else: 
        
        # load the studies ID from train, test and val 
        train = [x.split('\t') for x in open('../input_files/sentara_train_studies.txt').readlines()]
        val = [x.split('\t') for x in open('../input_files/sentara_val_studies.txt').readlines()]
        test = [x.split('\t') for x in open('../input_files/sentara_test_studies.txt').readlines()]
        
        studies_train = [item[1] for item in train[1:]]
        studies_val = [item[1] for item in val[1:]]
        studies_test = [item[1] for item in test[1:]]
       
        # split each set based on their study_ids
        train = sen_data[sen_data['study_id'].isin(studies_train)]
        test = sen_data[sen_data['study_id'].isin(studies_test)]
        val = sen_data[sen_data['study_id'].isin(studies_val)]
        
        y_train = train[[x for x in sen_data.columns if x.startswith('outcome_cancer_')]]
        x_train = train.drop(columns=[x for x in sen_data.columns if x.startswith('outcome_')])

        y_val = val[[x for x in sen_data.columns if x.startswith('outcome_cancer_')]]
        x_val = val.drop(columns=[x for x in sen_data.columns if x.startswith('outcome_')])


        y_test = test[[x for x in sen_data.columns if x.startswith('outcome_cancer_')]]
        x_test = test.drop(columns=[x for x in sen_data.columns if x.startswith('outcome_')])

        pkl.dump((x_train, y_train), open(train_path, 'wb')) 
        pkl.dump((x_val, y_val), open(val_path, 'wb')) 
        pkl.dump((x_test, y_test), open(test_path, 'wb'))
        
    print('Number of samples in train: %d, val: %d and test: %d' %(x_train.shape[0], x_val.shape[0], x_test.shape[0]))
        
    return x_train, y_train, x_val, y_val, x_test, y_test 


def split_maccabi(mac_data: pd.DataFrame, train_path = '../pkls/maccabi_train.pkl', val_path = '../pkls/maccabi_val.pkl', 
                  test_path = '../pkls/maccabi_test.pkl', overwrite = False):
    """
    Splits Maccabi dataframe into train, validation and test sets based on the patient ids. 
    If pickles already exist and there's no need to overwrite, load directly from pickles.
    Args:
    mac_data: maccabi dataframe to split.
    train_path (str): path to train set
    val_path (str): path to validation set
    test_path (str): path to test set
    overwrite (boolean): whether to overwrite existing files.
    Returns (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series): matrix (x) and labels (y) for train, 
    vat, test, respectively 
    """
    
    if os.path.isfile(train_path) and not overwrite: 
        x_train, y_train = pkl.load(open(train_path, 'rb')) 
        x_val, y_val = pkl.load(open(val_path, 'rb')) 
        x_test, y_test = pkl.load(open(test_path, 'rb')) 
    else: 
        
        # load the studies ID from train, test and val 
        train = pd.read_csv('../input_files/maccabi_train_studies.csv')
        val = pd.read_csv('../input_files/maccabi_val_studies.csv')
        ho = pd.read_csv('../input_files/maccabi_heldout_studies.csv')
        
        pats_train = train.patient_id.unique().tolist()
        pats_val = val.patient_id.unique().tolist()
        pats_test = ho.patient_id.unique().tolist()
       
        # split each set based on their patient ids
        train = mac_data[np.in1d(mac_data.index.get_level_values(0), pats_train)]
        test = mac_data[np.in1d(mac_data.index.get_level_values(0), pats_test)]
        val = mac_data[np.in1d(mac_data.index.get_level_values(0), pats_val)]
        
        y_train = train[[x for x in mac_data.columns if x.startswith('outcome_cancer_type')]]
        x_train = train.drop(columns=[x for x in mac_data.columns if x.startswith('outcome_')])

        y_val = val[[x for x in mac_data.columns if x.startswith('outcome_cancer_type')]]
        x_val = val.drop(columns=[x for x in mac_data.columns if x.startswith('outcome_')])


        y_test = test[[x for x in mac_data.columns if x.startswith('outcome_cancer_type')]]
        x_test = test.drop(columns=[x for x in mac_data.columns if x.startswith('outcome_')])

        pkl.dump((x_train, y_train), open(train_path, 'wb')) 
        pkl.dump((x_val, y_val), open(val_path, 'wb')) 
        pkl.dump((x_test, y_test), open(test_path, 'wb'))
        
    print('Number of samples in train: %d, val: %d and test: %d' %(x_train.shape[0], x_val.shape[0], x_test.shape[0]))
        
    return x_train, y_train, x_val, y_val, x_test, y_test 

    
def add_density_estimation_sentara(df, model_path='../pkls/breast_density_pkls/XGBoost/model.pkl', 
                    features_path='../pkls/breast_density_pkls/XGBoost/features.pkl'): 
        
    """ 
    Args: df_list (list) 
    model_path (str)
    features_path (str)
    Returns (pd.DataFrame, pd.DataFrame, pd.DataFrame) 
    """

    if os.path.isfile(model_path) and os.path.isfile(features_path): 
        model_density = pkl.load(open(model_path, 'rb')) 
        density_features = pkl.load(open(features_path, 'rb'))
        

        df['calc_breast_density_current'] = model_density.predict_proba(df[[x for x in density_features]])[:, 1] 
        
        return df
    
    else: 
            raise Exception('problem with model or features path')
            
            
def add_density_estimation_maccabi(df, model_path='../pkls/breast_density_pkls/XGBoost/model.pkl', 
                    features_path='../pkls/breast_density_pkls/XGBoost/features.pkl'): 
        
    """ 
    Args: df_list (list) 
    model_path (str)
    features_path (str)
    Returns (pd.DataFrame, pd.DataFrame, pd.DataFrame) 
    """

    if os.path.isfile(model_path) and os.path.isfile(features_path): 
        model_density = pkl.load(open(model_path, 'rb')) 
        density_features = pkl.load(open(features_path, 'rb'))
        

        df['calc_breast_density_current'] = model_density.predict_proba(df[[x for x in density_features]])[:, 1] 
        
        return df
    
    else: 
            raise Exception('problem with model or features path')
            
            
def add_bmi_estimation_sentara(df, model_path1= '../pkls/bmi_estimation_pkls/XGBoostRegressor/model_bmi_estimation_with_history.pkl',
                               model_path2= '../pkls/bmi_estimation_pkls/XGBoostRegressor/model_bmi_estimation_without_history.pkl', 
                               feats_wo_hist_path = '../pkls/bmi_estimation_pkls/XGBoostRegressor/features_without_history.pkl',
                       feats_w_hist_path = '../pkls/bmi_estimation_pkls/XGBoostRegressor/features_with_history.pkl'):
        
    """ 
    Args: df (Dataframe) 
    model_path (str)
    features_path (str)
    Returns (pd.DataFrame, pd.DataFrame, pd.DataFrame) 
    """


    if os.path.isfile(model_path1) and os.path.isfile(model_path2) and os.path.isfile(feats_wo_hist_path) and\
    os.path.isfile(feats_w_hist_path): 
        
        model_bmi_w_hist = pkl.load(open(model_path1, 'rb')) 
        model_bmi_wo_hist = pkl.load(open(model_path2, 'rb'))  
        features_w_hist = pkl.load(open(feats_w_hist_path, 'rb'))
        features_wo_hist = pkl.load(open(feats_wo_hist_path, 'rb'))

        df_w_prev_bmi = df[(df['bmi_max'].notna()) | (df['bmi_last'].notna())].copy()
        
        cat_feats = [x for x in df_w_prev_bmi.columns if 'ind' in x] + [x for x in df_w_prev_bmi.columns if 'cnt' in x] +\
              ['breast_density_past'] +['race'] + ['religion']

        # Fill missing values of categorical data with most frequent value
        df_w_prev_bmi.loc[:,cat_feats] = df_w_prev_bmi.loc[:,cat_feats].fillna(df_w_prev_bmi.loc[:,cat_feats].mode().iloc[0])
        
        df_w_prev_bmi['calc_bmi_current'] = model_bmi_w_hist.predict(df_w_prev_bmi[[x for x in features_w_hist]])
        

        df_wo_prev_bmi = df[(df['bmi_max'].isna()) & (df['bmi_last'].isna())].copy()
        df_w_prev_bmi.loc[:,cat_feats] = df_w_prev_bmi.loc[:,cat_feats].fillna(df_w_prev_bmi.loc[:,cat_feats].mode().iloc[0])
        df_wo_prev_bmi['calc_bmi_current'] = model_bmi_wo_hist.predict(df_wo_prev_bmi[[x for x in features_wo_hist]]) 

        frames = [df_w_prev_bmi, df_wo_prev_bmi]

        df_final = pd.concat(frames)

        return df_final 
    else: 
        raise Exception('problem with model or features path')    
        
def add_bmi_estimation_maccabi(df, model_hist_path= '../pkls/bmi_estimation_pkls/RandomForestRegressor/model_bmi_estimation_with_history.pkl',
                               model_no_hist_path= '../pkls/bmi_estimation_pkls/RandomForestRegressor/model_bmi_estimation_without_history.pkl', 
                               feats_wo_hist_path = '../pkls/bmi_estimation_pkls/RandomForestRegressor/features_without_history.pkl',
                       feats_w_hist_path = '../pkls/bmi_estimation_pkls/RandomForestRegressor/features_with_history.pkl'):
        
    """ 
    Args: df (Dataframe) 
    model_path (str)
    features_path (str)
    Returns (pd.DataFrame, pd.DataFrame, pd.DataFrame) 
    """


    if os.path.isfile(model_hist_path) and os.path.isfile(model_no_hist_path) and os.path.isfile(feats_wo_hist_path) and\
    os.path.isfile(feats_w_hist_path): 
        
        model_bmi_w_hist = pkl.load(open(model_hist_path, 'rb')) 
        model_bmi_wo_hist = pkl.load(open(model_no_hist_path, 'rb'))  
        features_w_hist = pkl.load(open(feats_w_hist_path, 'rb'))
        features_wo_hist = pkl.load(open(feats_wo_hist_path, 'rb'))

        df_w_prev_bmi = df[(df['bmi_max'].notna()) | (df['bmi_last'].notna())].copy()
        
        cat_feats = [x for x in df_w_prev_bmi.columns if 'ind' in x] + [x for x in df_w_prev_bmi.columns if 'cnt' in x] +\
              ['breast_density_past'] + ['past_birads_max']

        # Fill missing values of categorical data with most frequent value
        df_w_prev_bmi.loc[:,cat_feats] = df_w_prev_bmi.loc[:,cat_feats].fillna(df_w_prev_bmi.loc[:,cat_feats].mode().iloc[0])
        
        df_w_prev_bmi['calc_bmi_current'] = model_bmi_w_hist.predict(df_w_prev_bmi[[x for x in features_w_hist]])
        

        df_wo_prev_bmi = df[(df['bmi_max'].isna()) & (df['bmi_last'].isna())].copy()
        df_w_prev_bmi.loc[:,cat_feats] = df_w_prev_bmi.loc[:,cat_feats].fillna(df_w_prev_bmi.loc[:,cat_feats].mode().iloc[0])
        df_wo_prev_bmi['calc_bmi_current'] = model_bmi_wo_hist.predict(df_wo_prev_bmi[[x for x in features_wo_hist]]) 

        frames = [df_w_prev_bmi, df_wo_prev_bmi]

        df_final = pd.concat(frames)

        return df_final 
    else: 
        raise Exception('problem with model or features path')    
            
            

def add_likelihood_obesity_estimation_sentara(df, model_path1= '../pkls/likelihood_estimation_pkls/XGBoost/model_with_history.pkl',
                               model_path2= '../pkls/likelihood_estimation_pkls/XGBoost/model_without_history.pkl',
                               features_wo_hist='../pkls/likelihood_estimation_pkls/XGBoost/features_without_history.pkl',
                               features_w_hist='../pkls/likelihood_estimation_pkls/XGBoost/features_with_history.pkl'): 
        
    """ 
    Args: df (Dataframe) 
    model_path (str)
    features_path (str)
    Returns (pd.DataFrame, pd.DataFrame, pd.DataFrame) 
    """

    if os.path.isfile(model_path1) and os.path.isfile(model_path2) and os.path.isfile(features_wo_hist) and \
    os.path.isfile(features_w_hist):
        
        model_ob_w_hist = pkl.load(open(model_path1, 'rb')) 
        model_ob_wo_hist = pkl.load(open(model_path2, 'rb'))  
        features_wo_hist = pkl.load(open(features_wo_hist, 'rb'))
        features_w_hist = pkl.load(open(features_w_hist, 'rb'))

        df_w_prev = df[(df['bmi_max'].notna()) | (df['bmi_last'].notna())].copy()
        df_w_prev['calc_likelihood_obesity'] = model_ob_w_hist.predict_proba(df_w_prev[[x for x in features_w_hist]])[:, 1]

        df_wo_prev = df[(df['bmi_max'].isna()) & (df['bmi_last'].isna())].copy()
        df_wo_prev['calc_likelihood_obesity'] = model_ob_wo_hist.predict_proba(df_wo_prev[[x for x in features_wo_hist]])[:, 1]

        frames = [df_w_prev, df_wo_prev]

        df_final = pd.concat(frames)

        return df_final 
    else: 
        raise Exception('problem with model or features path') 
        
def add_likelihood_obesity_estimation_maccabi(df, model_path1= '../pkls/likelihood_estimation_pkls/XGBoost/model_with_history.pkl',
                               model_path2= '../pkls/likelihood_estimation_pkls/XGBoost/model_without_history.pkl',
                               features_wo_hist='../pkls/likelihood_estimation_pkls/XGBoost/features_without_history.pkl',
                               features_w_hist='../pkls/likelihood_estimation_pkls/XGBoost/features_with_history.pkl'): 
        
    """ 
    Args: df (Dataframe) 
    model_path (str)
    features_path (str)
    Returns (pd.DataFrame, pd.DataFrame, pd.DataFrame) 
    """

    if os.path.isfile(model_path1) and os.path.isfile(model_path2) and os.path.isfile(features_wo_hist) and \
    os.path.isfile(features_w_hist):
        
        model_ob_w_hist = pkl.load(open(model_path1, 'rb')) 
        model_ob_wo_hist = pkl.load(open(model_path2, 'rb'))  
        features_wo_hist = pkl.load(open(features_wo_hist, 'rb'))
        features_w_hist = pkl.load(open(features_w_hist, 'rb'))

        df_w_prev = df[(df['bmi_max'].notna()) | (df['bmi_last'].notna())].copy()
        df_w_prev['calc_likelihood_obesity'] = model_ob_w_hist.predict_proba(df_w_prev[[x for x in features_w_hist]])[:, 1]

        df_wo_prev = df[(df['bmi_max'].isna()) & (df['bmi_last'].isna())].copy()
        df_wo_prev['calc_likelihood_obesity'] = model_ob_wo_hist.predict_proba(df_wo_prev[[x for x in features_wo_hist]])[:, 1]

        frames = [df_w_prev, df_wo_prev]

        df_final = pd.concat(frames)

        return df_final 
    else: 
        raise Exception('problem with model or features path') 
        

def add_bmi_ground_truth_maccabi(df, bmis , D=100):
    
    """ 
    Args: Maccabi dataframe we want to add the BMI for. BMI values were taken from homer DB already filtered on patients we use
    for the Virtual Biopsy.
    bmis (df): dataframe where BMI values reside. Needs to be in the homer DB structure: patient_id, study_date, bmi
    D: int: Number of max days we want the BMI annotation to be apart from the study date.
    Returns pd.DataFrame with BMI values (only BMI annotation D days apart from study date)
    """
    
    pats = df.index.get_level_values(1).tolist()

    all_studies = []
    all_bmis = []
    all_closest_day_of_bmi_to_exam = []
    study_without_bmi_index = []


    for i, pat in enumerate(pats):

        print('Assigning BMI for patient {} out of {} \r'.format(i+1, len(pats)), end='', flush=True)

        df_pat = df[np.in1d(df.index.get_level_values(1), pat)]

        studies = df_pat.study_tag.tolist()
        
        if len(studies) >1: #correct code to run when there's more than 1 study
            continue

        bmi_closest_day = []
        closest_day_of_bmi_to_exam = []

        for s in studies:

            study_date = df_pat[df_pat['study_tag'] == s].index.get_level_values(2)[0]
            dates_bmi = bmis[bmis['patient_id'] == pat].study_date.tolist()

            if not dates_bmi:
                study_without_bmi_index.append(i) #append index of study without bmi
                continue

            diff_days = []

            for bmi_date in dates_bmi:

                diff_days.append(abs((study_date - datetime.strptime(bmi_date, '%Y-%m-%d %H:%M:%S'))).days)

            bmi_closest_day.append(bmis[bmis['patient_id'] == pat].iloc[diff_days.index(min(diff_days))].bmi)
            closest_day_of_bmi_to_exam.append(min(diff_days))

        all_closest_day_of_bmi_to_exam.append(min(diff_days))
        all_studies.append(studies)
        all_bmis.append(bmi_closest_day)

        all_studies_flat = [item for sublist in all_studies for item in sublist]
        all_bmis_flat = [item for sublist in all_bmis for item in sublist]
    #     all_closest_days_flat = [item for sublist in all_closest_day_of_bmi_to_exam for item in sublist]


    # remove studies without bmi from list of studies
    for i in study_without_bmi_index:
        del all_closest_day_of_bmi_to_exam[i]
        del all_studies_flat[i]

    bmi_data = {'study_tag':all_studies_flat, 'bmi_current': all_bmis_flat, 'NumberOfDaysFromStudyToBMI': all_closest_day_of_bmi_to_exam}

    df_bmi = pd.DataFrame(data = bmi_data)
    df_bmi = df_bmi[df_bmi['NumberOfDaysFromStudyToBMI'] < D]
    
    # MErge BMI to mac_data
    df_final = df.reset_index(level=[1,2]).merge(df_bmi[['study_tag', 'bmi_current']], how='left', on = ['study_tag']).set_index(['patient_id', 'study_date'])

    return df_final


def compute_and_add_avg_dicom_tags(df, dicom):
    """ 
    Args: Maccabi dataframe we want to add the dicom for. dicom tags were taken from homer DB already filtered on studies we use
    for the Virtual Biopsy. Computes avg dicom per CC and MLO.
    dicom_path (str)
    Returns pd.DataFrame with BMI values (only BMI annotation D days apart from study date)
    """
    
    # drop cases where we dont know study id
    dicom = dicom[~dicom.study_id.str.startswith('@')] 

    # Add a "study_tag" that is made of patient_id@study_id to make sure it's unique
    dicom['study_id'] = [i.split('@')[0] for i in dicom.study_id.tolist()]  #the first part before @ is the study_id itself
    dicom['study_tag'] = dicom['patient_id'].astype(str) + '@' + dicom['study_id'].astype(str)
    
    tags = ['BodyPartThickness', 'XRayTubeCurrent', 'KVP', 'ExposureTime',
       'DistanceSourceToDetector', 'DistanceSourceToPatient']

    studies = dicom.study_tag.unique().tolist()
    avg_cc = []
    avg_mlo = []

    for i, s in enumerate(studies):

        print('Processing {} out of {} studies\r'.format(i+1, len(studies)), end='', flush=True)

        df_temp = dicom[dicom['study_tag'] == s]
        avg_cc.append(df_temp[df_temp['SeriesDescription'].str.contains('CC')][tags].mean(axis=0).tolist())
        avg_mlo.append(df_temp[df_temp['SeriesDescription'].str.contains('MLO')][tags].mean(axis=0).tolist())

    avg_cc_array = np.array([item for sublist in avg_cc for item in sublist]).reshape(len(studies), len(tags))
    avg_mlo_array = np.array([item for sublist in avg_mlo for item in sublist]).reshape(len(studies), len(tags))


    d = {'study_tag': studies, 'BodyPartThickness_AVG_CC': avg_cc_array[:, 0],
        'XRayTubeCurrent_AVC_CC': avg_cc_array[:, 1], 'KVP_AVG_CC': avg_cc_array[:, 2],
        'ExposureTime_AVG_CC': avg_cc_array[:, 3], 'DistanceSourceToDetector_AVG_CC': avg_cc_array[:, 4],
        'DistanceSourceToPatient_AVG_CC': avg_cc_array[:, 5],

       'BodyPartThickness_AVG_MLO': avg_mlo_array[:, 0],
        'XRayTubeCurrent_AVC_MLO': avg_mlo_array[:, 1], 'KVP_AVG_MLO': avg_mlo_array[:, 2],
        'ExposureTime_AVG_MLO': avg_mlo_array[:, 3], 'DistanceSourceToDetector_AVG_MLO': avg_mlo_array[:, 4],
        'DistanceSourceToPatient_AVG_MLO': avg_mlo_array[:, 5] }  

    dftags = pd.DataFrame(data=d)
    
    # Merge
    df_final = df.reset_index(level=[0,1]).merge(dftags, how='left', on = ['study_tag']).set_index(['patient_id', 'study_date'])


    return df_final


def get_change_bmi_and_months_to_first_mg_sentara(df):
    """ 
    Args: df dataframe to which we want to add features
        mammo: mammography table taken from homer db table study_statuses filtered by MG exams
        bmi_table: bmi tabe with patient id and dates
    Returns pd.DataFrame with study_id, patient_id, study_date, change_bmi, months_to_first_mg, bmi_variance
    """
    
    bmi = pd.read_csv('../input_files/sentara_bmis_all.csv', index_col = 0) #bmi with dates
    mammo = mammo = pd.read_csv('../input_files/mammography_studies_new.txt', sep='\t') # all MG exams from study statuses table
    
    # drop duplicates:
    mammo.drop_duplicates(inplace=True)
    bmi.drop_duplicates(inplace=True)

    # filter bmi values:
    bmi = bmi[bmi['bmi'].between(10,65)]
    # change column name:
    bmi.rename(columns={'date': 'study_date'}, inplace=True)

    # make datetime object:
    mammo['study_date'] = pd.to_datetime(mammo['study_date']) 
    bmi['study_date'] = pd.to_datetime(bmi['study_date'])
    df['study_date'] = pd.to_datetime(df['study_date'])
    
    print('merging')
    
    # merge bmi and mammo table to sentara:
    df = pd.merge(df, bmi, how='left', on='patient_id', suffixes=('', '_bmi'))
    df = pd.merge(df, mammo, how='left', on='patient_id', suffixes=('', '_mammo'))
    
    print('deleing')
    # delete cases where we have bmis in the future, later than the study date
    df = df[df['study_date']>= df['study_date_bmi']]
    
    # make grouped object:
    grouped = df.groupby('patient_id')
    
    print('get bmi')
    # get bmi values per patient:
    df['bmi_variance'] = grouped['bmi'].transform('var')
    df['bmi_change'] = grouped['bmi'].transform(lambda x: (x.max())/x.min())

    # get month since first mammography:
    df['date_mammo_min'] = grouped['study_date_mammo'].transform('min')
    df['months_since_first_MG'] = (df['study_date'] - df['date_mammo_min']) / np.timedelta64(1, 'M')
    df['months_since_first_MG'] = df['months_since_first_MG'].round(0)
    
    #  column months since first mg will have negative values, because the study statuses only had future events.
    # set these values to nan

    idx_negative_months = df[df['months_since_first_MG']<0].index.tolist()
    df.loc[idx_negative_months, 'months_since_first_MG'] = df.loc[idx_negative_months, 
                'months_since_first_MG'].replace(df.loc[idx_negative_months, 'months_since_first_MG'].values, np.nan)
    
    # drop columns from merged tables:
    df.drop(columns=['modality', 'study_date_mammo', 'study_id_mammo', 'date_mammo_min', 'study_date_bmi', 'bmi'], inplace=True)
    
    features_df = df.drop_duplicates().reset_index(drop=True)[['patient_id', 'study_id', 'bmi_variance', 'bmi_change', 'months_since_first_MG' ]]
    
    return features_df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
