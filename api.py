from flask import Flask,request
from flask_cors import CORS
from cut_mode import CutOrder,word2index
import config as cfg
from loguru import logger
from predict_sentence import Model

class Modelconfig:
    def __init__(self,model_dir,batch_size) -> None:
        self.model_dir = model_dir
        self.batch_size = batch_size

def get_config(cfg):
    intent_cfg = Modelconfig(cfg.intent_model_dir,cfg.intent_batch_size)
    sender_cfg = Modelconfig(cfg.sender_model_dir,cfg.sender_batch_size)
    return intent_cfg,sender_cfg

app = Flask(__name__)

cuter = CutOrder('dict/jiebaword.txt')
intent_cfg,sender_cfg = get_config(cfg)
nlu_model = Model(intent_cfg)
sender_model = Model(sender_cfg)
CORS(app, resources=r'/*')

@app.route('/order',methods=['POST'])
def process_order():
    input_item = request.get_json()
    input_order = input_item['input_order']
    order_item = {
        'order':input_order,
        'intent':[],
        'label':[],
        "context": "",
        "phase": "",
        "sender": "",
        "task": "",
        'id':-1,

    }
    jsonitem = order_nlu(order_item,0)

    return jsonitem

def get_slots(slots):
    res = []
    num = 0
    for word,BIOkey in slots:
        
        temp = {
                "BIO": BIOkey.split('-')[-1],
                "ids": [
                    num,
                    num+len(word)-1
                ],
                "word": word
            }
        num = num+len(word)
        res.append(temp)
    return res

def get_intent(intent):
    if '#' in intent:
        res = intent.split('#')
    else:
        res = [intent]
    return res

def order_nlu(item,mode):
    order = item['order']
    intent = item['intent']
    label = item['label']
    if mode == 0:
        if len(label) == 0:
            cut_str = cuter.cut_order(order)
            newstr = ' '.join(cut_str)
            res = nlu_model.predict(newstr)
            sender = sender_model.predict(newstr)['intent']
            if len(intent) == 0:
                intent = get_intent(res['intent'])
            candidates = get_slots(res['slots'])
        else:
            candidates = label
    else:
        if mode == 1:
            wordlist = cuter.cut_order(order)
        elif mode == 2:
            wordlist = cuter.cut_order_word(order)
        else:
            wordlist = cuter.cut_order_word(order)
        
        index = word2index(wordlist)
        candidates = []
        for w,i in zip(wordlist,index):
            temp = {
                "word":w,
                "ids":i,
                'BIO':'O'
            }
            candidates.append(temp)

    jsontemp = {
        "task":item['task'],
        "id":item['id'],
        "order":order,
        "intent":intent,
        "sender":sender,
        'phase':item['phase'] if 'phase' in item else "",
        'context':item['context'] if 'context' in item else "",
        "candidates":candidates
    }

    return jsontemp



if __name__=='__main__':
    app.run(host='0.0.0.0',port=18080,debug=True)