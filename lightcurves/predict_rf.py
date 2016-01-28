import fnmatch
import os
import pickle
import pandas as pd

def run(field_num):
    rf_path = '/Users/cristobal/Documents/repos/msc-personal/lightcurves/rf_2class'
    clf = pickle.load(open(rf_path))

    field = 'F_'+str(field_num)
    print field
    data_path = '/Volumes/bernardita/MACHO_LMC/'+field
    files = os.listdir(data_path)
    final_df = pd.DataFrame()

    for f in files:
        if fnmatch.fnmatch(f, "*_ASCII.B.dat"):
            print f
            df = pd.read_table(data_path+"/"+f, sep='\s+')
            if df.empty or df.shape[0] == 0:
                continue
            df = df.fillna(0)
            macho_ids = df['#MACHO_ID']
            del df['#MACHO_ID']

            res = clf.predict_proba(df)

            prob_df = pd.DataFrame(res,columns=['prob1','prob2'])
            prob_df['macho_id'] = macho_ids
            final_df = final_df.append(prob_df)

    if final_df.empty is False:
        final_df = final_df.sort(columns=['prob1'], ascending=False)
    final_df.to_csv(field+'.csv', index=False)
