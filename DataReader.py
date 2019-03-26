# coding=utf-8
import pandas as pd

"""
transform DataFrame(np.array) to ffm data form

"""


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dftrain=None, dftest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dftrain is None)), "At least one is not None"
        assert not ((trainfile is not None) and (dftrain is not None)), "only one not None"
        assert not ((testfile is None) and (dftest is None)), "At least one is not None"
        assert not ((testfile is not None) and (dftest is not None)), "only one not None"

        self.trainfile = trainfile
        self.testfile = testfile
        self.dftrain = dftrain
        self.dftest = dftest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dftrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dftrain

        if self.dftest is None:
            dfTest = pd.read_csv(self.dftest)
        else:
            dfTest = self.dftest
        df = pd.concat([dfTrain, dfTest])

        self.feat_dict = {}
        fs = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = fs
                fs += 1
            else:
                fu = df[col].unique()
                self.feat_dict[col] = dict(map(fu, range(fs, len(fu) + fs)))
                fs += len(fu)
        self.feat_dim = fs


class Dataparser(object):

    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "At least one is not None"
        assert not ((infile is not None) and (df is not None)), "Just one is needed "

        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi['target'].values.tolist()
            dfi = dfi.drop(label, axis=1)
        # dfi is feature index
        # dfv is feature value(just can be binary(0/1) or float)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi = dfi.drop(col, axis=1)
                dfv = dfv.drop(col, axis=1)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1
        xi = dfi.values.tolist()
        xv = dfv.values.tolist()

        if has_label:
            return xi, xv, y
        else:
            return xi, xv


if __name__ == '__main__':
    feat_dict = FeatureDictionary(dfTrain=dfTrain,  # train
                                  dfTest=dfTest,  # test
                                  numeric_cols=config.NUMERIC_COLS,  # num
                                  ignore_cols=config.IGNORE_COLS)  # ignore
    data_parser = DataParser(feat_dict=feat_dict)
    # Xi_train ：列的序号
    # Xv_train ：列的对应的值
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)  # list
    Xi_valid, Xv_valid, y_valid = data_parser.parse(df=dfTest, has_label=True)
