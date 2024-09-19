from spmf import Spmf
import pandas as pd


db = [
    [[0], [1], [2], [3], [4]],
    [[1], [1], [1], [3], [4]],
    [[2], [1], [2], [2], [0]],
    [[1], [1], [1], [2], [2]],
]

def to_pandas_dataframe_with_SIDs(self, pickle=False):
    """
    Convert output to pandas DataFrame
    pickle: Save as serialized pickle
    """
    # TODO: Optional parameter for pickle file name

    if not self.patterns_:
        self.parse_output()

    patterns_dict_list = []
    for pattern_sup in self.patterns_:
        pattern = pattern_sup[:-1]
        sup = pattern_sup[-1:][0]
        sup = sup.strip()
        if not sup.startswith("#SUP"):
            print("support extraction failed")
        sup = sup.split()
        index = sup.index("#SID:")
        sids = sup[index+1:]
        sids = [int(id) for id in sids]
        sup = sup[1]

        patterns_dict_list.append({'pattern': pattern, 'sup': int(sup), 'sID': sids})

    df = pd.DataFrame(patterns_dict_list)
    self.df_ = df

    if pickle:
        df.to_pickle(self.output_.replace(".txt", ".pkl"))
    return df

Spmf.to_pandas_dataframe_with_SIDs = to_pandas_dataframe_with_SIDs
spmf = Spmf("PrefixSpan", spmf_bin_location_dir='/Users/rk3570/Desktop/RuleGraph', input_direct=db,
            output_filename="output.txt",
            arguments=[0, "2", True])

spmf.run()
# print(spmf.parse_output())
df = spmf.to_pandas_dataframe_with_SIDs()
print(df)