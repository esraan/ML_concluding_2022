import pandas as pd
import os
from scipy import stats
import scikit_posthocs as sp


if __name__ == "__main__":
    group_mrmr =pd.DataFrame(None)
    group_reliefF = pd.DataFrame(None)
    group_SelectFdr = pd.DataFrame(None)
    group_RFE = pd.DataFrame(None)
    group_hybrid_DReductionC = pd.DataFrame(None)
    group_GBC = pd.DataFrame(None)
    group_GBC_new = pd.DataFrame(None)

    main_path = "/Users/esraan/Desktop/ML_4/data/clean/all_together/data/"# "" /sise/home/esraan/ML_4/input/
    for dataset in os.listdir(main_path):
        os.chdir(main_path)
        if dataset.endswith("_result.csv"):
            df = pd.read_csv(dataset, low_memory=False)
            group_mrmr = pd.concat([group_mrmr,df.loc[df['param_reduce_dim__score_func'].str.contains('mrmr') ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)
            group_reliefF = pd.concat([group_reliefF,df.loc[df['param_reduce_dim__score_func'].str.contains('reliefF') ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)
            group_SelectFdr = pd.concat([group_SelectFdr,df.loc[df['param_reduce_dim__score_func'].str.contains('SelectFdr') ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)
            group_RFE =pd.concat([group_RFE,df.loc[df['param_reduce_dim__score_func'].str.contains('RFE') ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)
            group_hybrid_DReductionC = pd.concat([group_hybrid_DReductionC,df.loc[df['param_reduce_dim__score_func'].str.contains('hybrid_DReductionC') ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)
            group_GBC = pd.concat([group_GBC,df.loc[df['param_reduce_dim__score_func'].str.contains("GBC'>$", regex = True) ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)
            group_GBC_new = pd.concat([group_GBC_new,df.loc[df['param_reduce_dim__score_func'].str.contains('GBC_new') ].filter(regex='^mean',axis=1).filter(regex='AUC$',axis=1)],axis=0)

    crit, pvalue = stats.friedmanchisquare(group_mrmr,group_reliefF,group_SelectFdr,group_RFE,group_hybrid_DReductionC,group_GBC,group_GBC_new)

    print ("statistic: " +str(crit)+"pvalue: "+ str(pvalue))
    if pvalue<10: #0.05
        print("post-hoc values")
        group_mrmr.fillna(value=0,inplace=True)
        group_reliefF.fillna(value=0,inplace=True)
        group_SelectFdr.fillna(value=0,inplace=True)
        group_RFE.fillna(value=0,inplace=True)
        group_hybrid_DReductionC.fillna(value=0,inplace=True)
        group_GBC.fillna(value=0,inplace=True)
        group_GBC_new.fillna(value=0,inplace=True)
        allgroups = pd.concat([group_mrmr.reset_index(drop=True), group_reliefF.reset_index(drop=True)],axis=1,ignore_index=True)
        allgroups = pd.concat([allgroups,group_SelectFdr.reset_index(drop=True)],axis=1,ignore_index=True)
        allgroups = pd.concat([allgroups, group_RFE.reset_index(drop=True)], axis=1,ignore_index=True)
        allgroups = pd.concat([allgroups, group_hybrid_DReductionC.reset_index(drop=True)], axis=1,ignore_index=True)
        allgroups = pd.concat([allgroups, group_GBC.reset_index(drop=True)], axis=1,ignore_index=True)
        allgroups = pd.concat([allgroups, group_GBC_new.reset_index(drop=True)], axis=1,ignore_index=True)
        allgroups = allgroups.melt(var_name='FS alogrithms', value_name='values')
        # pick up test
        #pv = sp.posthoc_dscf([group_mrmr.values, group_reliefF.values, group_SelectFdr.values, group_RFE.values, group_hybrid_DReductionC.values, group_GBC.values,group_GBC_new.values])
        pv = sp.posthoc_conover(allgroups, val_col='values', group_col='FS alogrithms')
        pv.columns = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']
        pv.index = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']

        # pv.to_csv("post-hoc_results.csv")
        #
    # sns.heatmap(data=pv)


