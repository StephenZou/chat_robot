# chat_robot
意图识别、槽值填充、对话生成

用多分类：
训练完成：训练集-loss=intent_acc=0.99, domain_acc=0.986, slot_acc=0.581。
验证集-loss 0.1938, intent_acc 0.9612, domain_acc 0.9380, slot_acc 0.5124。

用CRF：
训练完成：训练集-loss=0.036, intent_acc=0.99, domain_acc=0.99, slot_acc=0.986。
验证集-loss 1.3665, intent_acc 0.9539, domain_acc 0.9574, slot_acc 0.7661。

可视化训练过程：tensorboard --logdir=./train/log/ --port 8123
