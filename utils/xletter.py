import sys
import numpy as np
sys.path.append('./')
from utils.param import FLAGS
class XletterPreprocessor():
    def __init__(self, xletter_dict, xletter_win_size):
        self.load_dict(xletter_dict)
        self.win_size = xletter_win_size
    def load_dict(self, filename):
        self.xlt_dict = {}
        idx = 1
        for line in open(filename):
            self.xlt_dict[line.strip()] = idx
            idx += 1
    def extract_xletter(self, term, xlt_dict):
        return [xlt_dict[term[i:i+3]] for i in range(0, len(term)-2) if term[i:i+3] in xlt_dict]
    def xletter_extractor(self, text):
        if isinstance(text,str):
            terms = text.strip().split(" ")
        else:
            terms = text.decode('utf-8').strip().split(" ")
        terms = ['#' + term + '#' for term in terms]
        terms_fea = [self.extract_xletter(term, self.xlt_dict) for term in terms]
        band = int(self.win_size / 2)
        offset = len(self.xlt_dict)
        res = ""
        for i in range(0, len(terms_fea)):
            tmp = ""
            for idx in range(0, self.win_size):
                if i - band + idx >= 0 and i - band + idx < len(terms_fea):
                    if len(tmp) and not tmp[-1] == ",":
                        tmp += ","
                    tmp += ",".join([str(int(idx*offset)+ix) for ix in terms_fea[i-band+idx]])
            if len(tmp):
                #res += ";" + tmp.strip(",") if len(res) else tmp.strip(",")
                res += ";" + tmp if len(res) else tmp
        return res
    def batch_xletter_extractor(self, text):
        indices = []
        value = []
        dense_shape = []
        maxLen = 0
        for batch_id in range(0,len(text)):
            if isinstance(text[batch_id], str):
                terms = text[batch_id].strip().split(" ")
            else:
                terms = text[batch_id].decode("utf-8").strip().split(" ")
            terms = ["#" + term + "#" for term in terms]
            terms_fea = [self.extract_xletter(term, self.xlt_dict) for term in terms]
            band = int(self.win_size / 2)
            offset = len(self.xlt_dict)
            maxLen = max(maxLen, len(terms_fea))
            for i in range(0, len(terms_fea)):
                for idx in range(0, self.win_size):
                    if i - band + idx >= 0 and i - band + idx < len(terms_fea):
                        newVal = [ix for ix in terms_fea[i-band+idx]]
                        indices.extend([[batch_id, i]]*len(newVal))
                        value.extend(newVal)
        dense_shape = [len(text),maxLen]
        #print(indices, value, dense_shape)
        return np.array(indices,dtype=np.int64), np.array(value,dtype=np.int32), np.array(dense_shape,dtype=np.int64)
