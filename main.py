import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from load_data import load_data, get_files_from_dir, root_path
import analyze
import describe

# ========================================================================= Data Loading and Treatment =========================================================================
df = load_data(get_files_from_dir(root_path))
df = pd.concat(df)

def clean_dataframe(df):
    df.set_index('Unnamed: 0', inplace=True)
    df.dropna()
    df.drop_duplicates()
    df['device_name'] = df['device_name'].astype('category')

clean_dataframe(df)


"""describe.shapes(df)
describe.boxplot_msm_size(df)
describe.describeDataProtocols(df)
describe.boxplot_sum_et(df)"""
"""describe.min_et(df)
describe.med_et(df)
describe.var(df)
describe.q1(df)
describe.q3(df)
describe.sum_e(df)
describe.max_e(df)
describe.average(df)"""

#describe.epoch_timestamp(df)     #ne marche pas (programme plante à l'éxécution -> trop d'éléments ?)
describe.most_freq_d_ip(df)
describe.most_freq_dport(df)
describe.most_freq_sport(df)



# ========================================================================= Describe =========================================================================
#describe.df_description(df)
#describe.shapes(df)
#describe.boxplot_msm_size(df)
#describe.describeDataProtocols(df)
#describe.boxplot_sum_et(df)
#describe.med_et(df)
#describe.var(df)
#describe.q1(df)
#describe.q3(df)
#describe.sum_e(df)
#describe.max_e(df)
#describe.average(df)

#describe.describeTotalLength(df)

# ========================================================================= Analysis =========================================================================
#analyze.correlation_heat_map(df)
#analyze.feature_importance(df)
analyze.pairplot_feature_importance(df) # Il faut grouper les devices par types et faire le pair plot, aussi masquer la moitié inutile
#analyze.kde_plot_mfreqdip(df)
#analyze.kde_plot_mfreqdport(df)
#analyze.kde_plot_mfreqsport(df)
#analyze.kde_plot_epochtimestamp(df)
#analyze.line_plot_arlobasecam_dip_ts(df)

