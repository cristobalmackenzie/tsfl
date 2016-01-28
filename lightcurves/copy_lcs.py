import pickle
import tarfile

var_lc_df = pickle.load(open('var_lc_df','r'))
current_tar_path = ""
tar = None

for m_id in var_lc_df['macho_id']:
    id_arr = m_id.split('.')
    tar_path = '/Volumes/bernardita/MACHO_LMC/F_'+id_arr[0]+'/'+id_arr[1]+'.tar'

    if tar_path != current_tar_path:
        current_tar_path = tar_path
        if tar != None:
            tar.close()
        tar = tarfile.open(tar_path)

    member_name = 'F_'+id_arr[0]+'/'+id_arr[1]+'/lc_'+m_id+'.B.mjd'
    tar.extract(tar.getmember(member_name),'var_lcs')
