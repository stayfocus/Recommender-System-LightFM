# -*- coding: utf-8 -*-

import os
import sys

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import tensorflow as tf
from networks import NetworkAlbertTextCNN
from classifier_utils import get_feature_test, id2label, DataProcessor
from hyperparameters import Hyperparamters as hp
import csv


class ModelAlbertTextCNN(object, ):
    """
    Load NetworkAlbert TextCNN model
    """

    def __init__(self):
        self.albert, self.sess = self.load_model()

    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert = NetworkAlbertTextCNN(is_training=False)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd, hp.file_load_model))
                print(checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert, sess


MODEL = ModelAlbertTextCNN()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's label.
    """
    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids: [feature[2]],
          }
    prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]
    # print(prediction)
    r = []
    for i in range(len(prediction)):
        if prediction[i] != 0.0 and i != 0:
            r.append(id2label(i))
    return r
    # return [id2label(l) for l in np.where(prediction == 1)[0] if l != 0]


if __name__ == '__main__':
    # content = ['在钱塘开茶铺的赵盼儿惊闻未婚夫、新科探花欧阳旭要另娶当朝高官之女，不甘命运的她誓要上京讨个公道。在途中她遇到了出自权门但生性正直的皇城司指挥顾千帆，并卷入江南一场大案，两人不打不相识从而结缘。赵盼儿凭借智慧解救了被骗婚而惨遭虐待的“江南第一琵琶高手”宋引章与被苛刻家人逼得离家出走的豪爽厨娘孙三娘，三位姐妹从此结伴同行，终抵汴京，见识世间繁华。为了不被另攀高枝的欧阳旭从汴京赶走，赵盼儿与宋引章、孙三娘一起历经艰辛，将小小茶坊一步步发展为汴京最大的酒楼，揭露了负心人的真面目，收获了各自的真挚感情和人生感悟，也为无数平凡女子推开了一扇平等救赎之门。',
    #            '《狮子山下的故事》以一家香港茶餐厅为载体，通过讲述几代人的生活故事，反映香港从1984年中英联合声明发表到2019年新中国成立70周年的时代变迁，展现了香港同胞与祖国同呼吸、共命运，在逆境中不畏风雨、团结奋斗、艰苦打拼的精神。',
    #            '全职太太方糖本过着幸福的婚姻生活，30岁生日后，她发现丈夫高家为一连串不合理行为，终于在一次意外中，确定丈夫有婚外情。经过内心的纠结和折磨，她选择冷静处理，希望丈夫能坦诚相对、迷途知返，但丈夫的躲避、欺瞒和不负责让她极度失望。同时，高家为的合伙人兼竞争对手齐妙，本想抓住高家为的婚姻危机来打压他，却弄巧成拙，让自己的事业受到牵连。齐妙忙于追求名利，早已疏离相识于微的丈夫，直到对方提出离婚才恍然大悟，即使自己拥有了物质与名气，却过得不开心。而齐妙的助理大雨忙于报复负心汉与教唆者，也忽略了家人。经历了各种变故，方糖选择走出舒适区，不再依赖婚姻，重返职场迎接新生活;齐妙终于明白人生最重要的是家人之间的关爱，学会平衡自己的生活和事业;而大雨则放下执念，人生不该困于无可挽留的过去，而是应当好好珍惜当下。',
    #            '见习警员李大为、夏洁、杨树和赵继伟在派出所的工作与生活，一开始对警察这个职业怀抱各种憧憬，然而当面对巨大的工作压力、尖锐的矛盾冲突以及自媒体时代给派出所工作带来的新挑战时，他们经历了困.惑与挫折，也有过迷茫、怀疑甚至放弃的念头，但在人民警察光荣传统的熏陶下，在老一辈的言传身教中，这4个年轻人经受住了考验，迅速成长，他们对自身的使命和荣誉有了新的认识，最终成长为合格的人民警察。',
    #            '该剧以“五四”新文化运动为背景，讲述了李大钊、陈独秀、胡适从相识、相知到分手，走上不同人生道路的传奇故事，展现了从新文化运动到中国共产党建立的光辉历程。',
    #            '天赋异禀的结巴少年“秦风”警校落榜，被姥姥遣送泰国找远房表舅——号称“唐人街第一神探 ”，实则“猥琐”大叔的“唐仁“散心。不想一夜花天酒地后，唐仁沦为离奇凶案嫌疑人，不得不和秦风亡命天涯，穷追不舍的警探——-“疯狗”黄兰登；无敌幸运的警察——“草包”坤泰；穷凶极恶、阴差阳错的“匪帮三人组”；高深莫测的“唐人街教父”；“美艳风骚老板娘”等悉数登场。七天，唐仁、秦风这对“欢喜冤家”、“天作之合”必须取长补短、同仇敌忾，他们要在躲避警察追捕、匪帮追杀、黑帮围剿的同时，在短短“七天”内，完成找到“失落的黄金”、查明“真凶”、为他们“洗清罪名”这些“逆天”的任务。',
    #            '异世界皇都，天神赤发鬼残暴统治，滥杀无辜。少年空文因被赤发鬼追杀，决定奋起反击。在黑甲的指引下，空文踏上了凡人弑神之路。这是小说家路空文笔下的奇幻世界。没想到小说的进程，竟然影响着现实世界。这时一名男子接下了刺杀他的任务。',
    #            '曾经在赛车界叱咤风云，却因非法飙车被禁赛五年的赛车手张驰如今只能经营炒饭大排档。年近四十的他决定重返车坛挑战年轻一代的天才林臻东，却遭遇重重障碍——没钱没车没队友，甚至驾照都得重新考。他找来曾经的搭档兼领航员孙宇强和昔日车队技师记星帮忙，好不容易凑齐了装备准备参赛，领航员孙宇强却又出了事故。',
    #            '影片以一战时期，1917年春某一天的两个小时为背景，当时德军从战场撤退到兴登堡防线，那里设置了地雷和狙击手作为陷阱。英国军队在与德军长期的僵局后，寻找机会消灭对手，计划进攻兴登堡防线，在最后关头才发觉这是德军的陷阱。布雷克和斯科菲尔德两位年轻的英军士兵奉命去前线传达命令，这次任务相当艰巨，不但要穿过前线和敌人阵地，留给他们完成任务的时间也只有一天，这不仅仅是为了拯救1600名英国士兵，而且布雷克的哥哥也在前线。',
    #            '2031年，人类试图阻止全球变暖的实验失败，极寒造成地球上绝大部分生命死亡。在冰河灾难中幸存下来的所有人登上了一辆如同诺亚方舟的列车，列车依靠永动机绕着地球不停行驶。在这列车厢有着等级之分的列车上，饱受饥饿之苦、生活在恶劣环境的末节车厢的人们在革命领袖柯蒂斯的带领下，为了生存一节车厢一节车厢的向前突进，掀起了一场向车头进军的“革命”。',
    #            '20世纪40年代的中国广东，有一名无可药救的小混混阿星，此人能言善道、擅耍嘴皮，但意志不坚，一事无成。他一心渴望加入手段冷酷无情、恶名昭彰的斧头帮，并梦想成为黑道响叮当的人物。此时斧头帮正倾全帮之力急欲铲平唯一未收入势力范围的地头，未料该地卧虎藏龙，痴肥的恶霸女房东肥婆加上与其成对比的懦弱丈夫二叔公，率领一班深藏不漏的武林高手，大展奇功异能，对抗恶势力。当黑帮大哥请来各界武林高手进攻猪笼城寨时，阿星趁机自荐，阿星不断向黑帮大哥 “表现”自己，最后决定去“杀人”，但是连拿刀子都会手颤的他又怎么杀人，他只能成为斧头帮在纤灭猪笼城寨时利用的工具，阿星没有想到后果会如此严重。',
    #            '富豪刘轩的地产计划涉及填海工程，威胁靠海为生的居民。因为人类对大海及生态的破坏，美人鱼只能被赶到了一艘破船里艰难生存，背负家族秘密的珊珊被派遣前往阻止填海计划。刘轩是一个靠自己努力才取得成就的人，虽然表面有钱但实则空虚寂寞的他和美人鱼珊珊在交手过程中互生情愫。刘轩最终因为爱上珊珊而停止填海工作，但珊珊却因意外受伤而消失于大海'
    #            ]
    content = [
        '电影围绕幸福是奋斗出来的主题，树立脱贫的自信心；通过精准扶贫与扶志扶智结合，摒弃“歇帮”和“等、靠、要”行为，助推贫困户和贫困村改变面貌，赢得幸福！'
                ]
    for i in content:
        print(get_label(i))

    # print(hp.dict_id2label)

    # csv_header = ['content', 'label_original', 'label_predict']  # csv表头
    # with open('textcnn_iqiyi_60_trans.csv', 'w', encoding='utf-8', newline='') as file_obj:
    #     writer = csv.writer(file_obj)  # 创建对象
    #     writer.writerow(csv_header)  # 写表头
    #
    # test_dir = os.path.join(hp.data_dir, hp.test_data)
    # for line in open(test_dir, 'r', encoding='utf-8'):  # 读取测试集数据
    #     label_original = line.strip().split(',', 1)[1].split(",")
    #     if len(label_original) > hp.label_length:  # 取出指定个数的标签
    #         label_original = label_original[len(label_original) - hp.label_length:]  # 切片操作
    #     content = line.strip()
    #     content = content.replace(str(label_original).strip('[').strip(']').replace('\'', '').replace(' ', ''),
    #                               '').rstrip(',')  # 取出对应的媒资简介
    #     label_original_trans = []  # 将原始标签从数字转换为具体标签
    #     for i, element in enumerate(label_original):
    #         if element == '1':
    #             label_original_trans.append(hp.dict_id2label[str(i)])
    #     label_predict = get_label(content)  # 使用简介进行标签预测
    #     # print('content:' + content)
    #     # print('label_original:' + str(label_original))
    #     # print('label_original_trans:' + str(label_original_trans))
    #     # print('label_predict:' + str(label_predict))
    #     with open('textcnn_iqiyi_60_trans.csv', 'a+', encoding='utf-8', newline='') as file_obj:
    #         csv_content = [content]
    #         csv_content.append(str(label_original_trans))
    #         csv_content.append(str(label_predict))
    #         writer = csv.writer(file_obj)
    #         writer.writerow(csv_content)

    # test_dir = os.path.join(hp.data_dir, hp.test_data)
    # for line in open(test_dir, 'r', encoding='utf-8'):  # 读取测试集数据
    #     content = line
    #     label_predict = get_label(content)  # 使用简介进行标签预测
    #     with open('textcnn_40000_series.csv', 'a+', encoding='utf-8', newline='') as file_obj:
    #         csv_content = [content]
    #         csv_content.append(str(label_predict))
    #         writer = csv.writer(file_obj)
    #         writer.writerow(csv_content)
