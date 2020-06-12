# emotion_detection_server
使用PaddlePaddle/models/emotion_detection样例代码封装的情感预测，数据为emotion_detection_textcnn-1.0.0.tar.gz保存所得

依赖库 paddlepaddle-gpu     1.8.0.post97 

参考： https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/emotion_detection

可执行项目： https://aistudio.baidu.com/aistudio/projectdetail/544264

启动GPU环境，
cd /home/aistudio/work 
!python emotion_sentence.py


??
3个线程以上同时进行预测，exe.run会返回错误结果或者空列表；2个线程同时预测，结果正常；



