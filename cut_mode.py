import jieba
import re

def word2index(wordlist):
    index = []
    start = 0
    for text in wordlist:
        end = len(text) + start
        temp = (start,end-1)
        index.append(temp)
        start = end
    return index

class CutOrder():
    def __init__(self,dict_path) -> None:
        jieba.load_userdict(dict_path)
    
    def cut_special_number(self,wordlist):
        m = re.compile(r'([洞幺两三四五六拐八九0-9])')
        res = []
        for word in wordlist:
            if re.search(m,word):
                temp = re.split(m,word)
                for t in temp:
                    if t != '':
                        res.append(t)
            else:
                res.append(word)
        return res

    def cut_order(self,order):
        seg = jieba.cut(order)
        seglist = [item for item in seg]
        seglist = self.cut_special_number(seglist)
        return seglist
    
    def cut_zh(self,wordlist):
        m = re.compile(r'([\u4e00-\u9fa5])')
        res = []
        for word in wordlist:
            if re.search(m,word):
                temp = re.split(m,word)
                for t in temp:
                    if t != '':
                        res.append(t)
            else:
                res.append(word)
        return res

    def cut_order_word(self,order):
        seg = jieba.cut(order)
        seglist = [item for item in seg]
        seglist = self.cut_zh(seglist)
        return seglist

if __name__=="__main__":
    text = '修正海压幺洞幺五hotelhotel幺幺到三洞左等待点海南拐洞两五'
    cutter = CutOrder('dict/jiebaword.txt')
    res = cutter.cut_order(text)
    index = word2index(res)
    print(res)
    print(index)

    # # 制作字典
    # with open('dict/chatword.txt','r',encoding='utf8') as fp:
    #     alltext = fp.readlines()
    # res = []
    # for text in alltext:
    #     word = text.strip()
    #     temp = word +' 10000\n'
    #     if temp not in res:
    #         res.append(temp)
    # with open('dict/jiebaword.txt','w+',encoding='utf8') as fr:
    #     for item in res:
    #         fr.write(item)