from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import utils
import threading
import time,random

"""
Emotion detection use paddlepaddle
"""
class PaddleEmotion(object):
    def __init__(self):        
        #paddle预测模型加载
        #self.dev_count = fluid.core.get_cuda_device_count()
        #self.place = fluid.CPUPlace()
        self.place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(self.place)

        [self.infer_program, self.feed_names, self.fetch_targets] = fluid.io.load_inference_model(
            dirname="./inference_model_textcnn",
            executor=self.exe,
            model_filename="model.pdmodel",
            params_filename="params.pdparams")
        self.vocab = utils.load_vocab("./vocab.txt")
        
    #预测结果，语句需要预先分词
    def emotion_list(self, sentencelist):
        data = []
        seq_lens = []		
        for sentence in sentencelist:
            wids = utils.query2vocab_ids(self.vocab, sentence)
            wids, seq_len = utils.pad_wid(wids)
            data.append(wids)
            seq_lens.append(seq_len)
        
        data = np.array(data)
        seq_lens = np.array(seq_lens)

        labelList = []
        pred = self.exe.run(self.infer_program,
                        feed={self.feed_names[0]: data,self.feed_names[1]: seq_lens},
                        fetch_list=self.fetch_targets,
                        return_numpy=True)
        if len(pred) != 0:
            for probs in pred[0]:
                #print("%d\t%f\t%f\t%f" % (np.argmax(probs), probs[0], probs[1], probs[2]))
                labelList.append(np.argmax(probs)) # 返回label, 0表示消极；1表示中性；2表示积极
        else:
            print("exe.run return None when emotion_list")
            labelList = [1 for x in range(len(sentencelist))] # 运算无结果时都默认为中性

        return labelList
    
def emotion_thread(emotion):
    num = 0
    while num < 100:
        word = emotion.emotion_list(["我 讨厌 你 ， 哼哼 哼 。 。", "这家 店铺 还 不错 ， 味道 非常 棒 ， 我 喜欢 麻婆豆腐 这个 菜"])
        print(word)
        num += 1
        #time.sleep(random.random()) # 将各线程随机休眠，错开算法执行时间可以提高线程数
        
        
if __name__ == "__main__":
    #emotion = PaddleEmotion()
    #word = emotion.emotion_list(["我 讨厌 你 ， 哼哼 哼 。 。", "这家 店铺 还 不错 ， 味道 非常 棒 ， 我 喜欢 麻婆豆腐 这个 菜"])
    #print(word)
    # 正常返回结果为 [0, 2]
    
    emotion = PaddleEmotion()
    # 3个线程以上同时进行预测，exe.run会返回错误结果或者空列表；2个线程同时预测，结果正常；
    thread_list = []
    for i in range(2):  # ?? 3
        t = threading.Thread(target=emotion_thread, args=(emotion,))
        thread_list.append(t)
    
    for t in thread_list:
        t.setDaemon(True)
        t.start()
    for t in thread_list:
        t.join()    



