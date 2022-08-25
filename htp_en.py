# 參考資料：https://www.i4k.xyz/article/weixin_44613063/108038602


import numpy as np
import time
import cv2
import os
import pathlib

# 取得目前檔案所在的資料夾 
SRC_PATH =  pathlib.Path(__file__).parent.absolute()
UPLOAD_FOLDER = os.path.join(SRC_PATH)

print(UPLOAD_FOLDER)

def detect(image_position):
    print(image_position)
    LABELS = open(UPLOAD_FOLDER+"/obj(1)/obj.names").read().strip().split("\n")
    np.random.seed(666)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    # 導入 YOLO 配置和權重文件並加載網路：
    net = cv2.dnn.readNetFromDarknet(UPLOAD_FOLDER+'/obj(1)/yolov4-tiny-obj.cfg', UPLOAD_FOLDER+'/obj(1)/yolov4-tiny-obj_best.weights')
    # 獲取 YOLO 未連接的輸出圖圖層
    layer = net.getUnconnectedOutLayersNames()
    image = cv2.imread(image_position)
    # /Users/wen/Downloads/59張畫/55.jpg
    # 獲取圖片尺寸
    (H, W) = image.shape[:2]
    # 從輸入圖像構造一個blob，然后執行 YOLO 對象檢測器的前向傳遞，給我们邊界盒和相關概率
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    # 前向傳遞，獲得信息
    layerOutputs = net.forward(layer)
    # 用於得出檢測時間
    end = time.time()
    print("YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []
    txtas=[]

    txt=''

    # 循環提取每個輸出層
    for output in layerOutputs:
        # 循環提取每個框
        for detection in output:
            # 提取當前目標的 類別ID(編號) 和 可信度
            scores = detection[5:]
            classID = np.argmax(scores)     # np.argmax(a) 找a陣列中的最大值
            confidence = scores[classID]

            # 通過確保檢測概率大於最小概率來過濾弱預測
            if confidence > 0.0:
                # 將邊界座標相對於圖像的大小進行縮放，YOLO 返回的是邊界框的中心(x,y)座標
                # 後面是邊界框的寬度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                txta=detection[0:4]
                (a, b, c, d) =txta.astype("float")

                # 轉換出邊框左上角座標
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # 更新邊界框座標、可信度和類ID的列表
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                txtas.append([a, b, c, d])
        
    # 非最大值抑制，確定唯一邊框，原始參數:idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, 0.3)
        
    
    door=0
    window=0
    chimney=0
    crown=0
    bark=0
    fruit=0
    face=0
    body=0
    neck=0
    house=0
    tree=0
    person=0
    classes=[door,window,chimney,crown,bark,fruit,face,body,neck,house,tree,person]
    hps=[]
    tps=[]
    pps=[]
    # 設定房樹人整體位置的初始值
    (x2,y2)=(H, W)
    (w2, h2)=(0,0)

    # 確定每個對象只少有一個框存在
    if len(idxs) > 0:
        # 循環畫出表存的邊框
        for i in idxs.flatten():
            # 提取座標和寬度
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # 畫出邊框和標籤
            color = [int(c) for c in COLORS[classIDs[i]]]
            #  cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度, 線條類型)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            #  cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, lineType=cv2.LINE_AA)

            for j in range(len(classes)):
                if LABELS[classIDs[i]]==LABELS[j]:
                    classes[j]=classes[j]+1
            # 自動標籤
            txt=txt+'%d'%classIDs[i]+' %f'%txtas[i][0]+' %f'%txtas[i][1]+' %f'%txtas[i][2]+' %f'%txtas[i][3]+'\n'

            # 取得房樹人中心座標
            
            if LABELS[classIDs[i]]=='house':
                hp=[x+(w/2),y+(h/2)]
                hps.append(hp)
                print('House:')
                print(hps)
            elif LABELS[classIDs[i]]=='tree':
                tp=[x+(w/2),y+(h/2)]
                tps.append(tp)
                print('Tree:')
                print(tps)
            elif LABELS[classIDs[i]]=='person':
                pp=[x+(w/2),y+(h/2)]
                pps.append(pp)
                print('Person:')
                print(pps)
        

        #print('\n'+txt) # 測試自動標註的檔案
        
        name=os.path.basename(image_position)
        name = name.split(".")[0] #去掉檔案格式，取得檔案原名稱
        
        file = open(UPLOAD_FOLDER+'/txt/'+name+'.txt','w+')
        file.write(txt)
        file.close()
            
        # 找出房樹人整體的位置
        # 循環畫出表存的邊框
        for j in idxs.flatten():
            print('%d/'%boxes[j][0]+'%d/'%boxes[j][1]+'%d/'%boxes[j][2]+'%d/'%boxes[j][3])
            # 提取座標和寬度
            (x2, y2) = (min(x2,boxes[j][0] ), min(y2,boxes[j][1] ))
            (w2, h2) = (max(w2,boxes[j][0]+boxes[j][2] ), max(h2,boxes[j][1]+boxes[j][3] ))

        #print('\n'+'%d/'%x2+'%d/'%y2+'%d/'%w2+'%d/'%h2) #測試all的邊框   
        # 畫出邊框和標籤
        color = [int(c) for c in COLORS[classIDs[i]]]
        #  cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度, 線條類型)
        cv2.rectangle(image, (x2, y2), (w2, h2), (0,0,255), 2, lineType=cv2.LINE_AA)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        #  cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
        cv2.putText(image, 'all', (x2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,255), 2, lineType=cv2.LINE_AA)

    

    # 分析辨識結果，並產出圖片描述跟代表意義
    result=''
    result_dc=''
    warn=''
    # classes=[0door,1window,2chimney,3crown,4bark,5fruit,6face,7body,8neck,9house,10tree,11person]
    # 確保有偵測到房樹人後再開始進行分析
    if classes[0]==0 and classes[0]==0 and classes[0]==0:
        result=result+'DETECT ERROR. Please re-enter the correct pictures of houses, trees, and people, or contact professionals for analysis.'
        result_dc=result_dc+'DETECT ERROR.'
    else:
        # 房的結果
        result=result+('  It can be seen from your picture,')
        if classes[0]>=1:   # 有門
            result=result+('you do not exclude contact with the outside world,') # 原：你個性可能比較外向，想要和他人接觸相處，也希望能找到能夠理解自己的人，不過相對起來也可能會比較依賴他人，
            result_dc=result_dc+('  The illustrator may be more outgoing, wants to communicate with others and be understood, and likes to rely on others;')
        elif classes[0]==0: # 沒門
            result=result+('your personality is more independent,') # 原：你比較保守，在和他人相處上也比較慢熟，比較喜歡和自己相處的時光，在家庭方面也是屬於比較獨立的人，不太依賴他人，
            result_dc=result_dc+('  The illustrator may be more defensive towards the outside world, dislikes others taking the initiative to approach them, and their personality is relatively cold and withdrawn, relationships are not deep;') 
        if classes[1]>=1:   # 有窗
            result=result+('like to make friends, ') # 原：不會排斥讓別人了解自己，也比較希望能和別人交流和溝通，
            result_dc=result_dc+('The illustrator has a willingness to communicate with the outside world;')
        elif classes[1]==0: # 沒窗
            result=result+('relatively cautious, ') # 原：在個性上也比較謹慎，不太會輕易下決定，在和他人相處上也是，比較不會主動去與陌生人交際，
            result_dc=result_dc+('The illustrator may be more withdrawn, may be more paranoid in getting along with others, and may also have a tendency to possess delusions of persecution, but a conclusion can only be made after careful evaluation, and it is only speculation at present;') 
            warn=warn+'May be prone to delusions of persecution!'
        if classes[2]>=1:   # 有煙囪
            result=result+('and values the family.') # 原：你關注家庭，會在意家庭上給予的溫暖和權力；
            result_dc=result_dc+('The illustrator may have been more concerned with the home and the warmth given by the home.\n')
        elif classes[2]==0: # 沒煙囪
            result=result+('and pursuit of family warmth.') # 原：最近你的情緒可能比較低落，在家庭方面可能有遇到一些問題，需要溫暖和幫助；
            result_dc=result_dc+('The illustrator may have been relatively passive recently and lacked psychological warmth in the family.\n')
            warn=warn+'Family relation may not be close!'
        # 樹的結果 3crown,4bark,5fruit
        result=result+('In addition,')
        if classes[3]>=1:   # 有樹冠
            result=result+('you are more positive,') # 原：你對未來可能有遠大的目標，比較積極上進，對自己也比較有信心，個性樂觀，
            result_dc=result_dc+('  The illustrator may have less self-confidence problems, and may have a certain degree of self-confidence in themselves and a desire to make progress;')
        elif classes[3]==0: # 沒樹冠
            result=result+('you are not keen on communication,') # 原：你個性比較內向，不是很擅長跟別人相處，對自己也比較沒有信心，
            result_dc=result_dc+('  The illustrator may have low self-esteem and depression, and the personality is relatively introverted and not able to communicate well with others. In addition, there is the possibility of suffering from schizophrenia, but it needs further understanding and can only be confirmed after diagnosis by an expert;')
            warn=warn+'Possibility of schizophrenia!'
        if classes[4]>=1:   # 有樹紋 
            result=result+('Observant person') # 原：在過去你可能有遇到一些問題和事件，在心中留下一些影響，且對事物可能比較敏銳，也比較有觀察力，
            result_dc=result_dc+('The illustrator may be more sensitive to the outside world in personality, and they may have encountered some things in the process of growing up, which may affect them;')
            warn=warn+'There may be some harm in growing up!'
        elif classes[4]==0: # 沒樹紋
            result=result+('Energetic person') # 原：在生活上你比較積極，很有活力和生命力，比較重視現實但也保有一些期望和想像力，
            result_dc=result_dc+('The tree in the picture is very simple and has no bark, indicating that the illustrator may not have encountered any difficulties in the past, or has gotten out of the difficulties, and is currently living a positive life;')
        if classes[5]>=1:   # 有果實
            result=result+(', expects a successful future.') # 原：近期你對自己和生活上可能有一些期待和想要達成的目標；
            result_dc=result_dc+('The illustrator may have some desires and goals.\n')
        elif classes[5]==0: # 沒果實
            result=result+('.') # 原：目前你可能沒有什麼想要達成的事情，比較謙虛不會自負，但對自己的發展比較沒有頭緒；
            result_dc=result_dc+('The illustrator may not have set achievable goals at present, and they may not have any requirements for themself at present.\n')
        # 人的結果 6face,7body,8neck
        result=result+('At the same time,')
        if classes[7]>=1:   # 有火柴人
            result=result # 原：你不太願意表露真實的自我，在圖畫中也有所掩飾，且可能對這項測驗比較有防禦性或拒絕的態度，
            result_dc=result_dc+('  The illustrator has the ability to cover up, has a strong ability to lie, may be reluctant to reveal their true self, and is defensive;')
        elif classes[7]==0: # 沒火柴人
            result=result # 原：你對此測驗有一定的信任程度，畫作中也沒有特別隱藏和掩飾，
            result_dc=result_dc+('  The illustrator has a certain degree of trust in this test, and they are less suspicious of the test, and they do not hide or cover up too much;')
        if classes[6]>=1:   # 有五官
            result=result+('you are easy to adapt to the environment,') # 原：你願意和外界接觸，也比較擅長適應新的環境，
            result_dc=result_dc+('Illustrators may be willing to engage with the outside world, and are less prone to maladaptation in new environments;')
        elif classes[6]==0: # 沒五官
            result=result+('you are shy,') # 原：你可能不太擅長應付交際方面的事務，個性比較害羞，進入新環境時需要相對較多的時間來適應，
            result_dc=result_dc+('Illustrators may prefer to avoid interpersonal relationships, are not able to adapt well to the environment, are more shy and more self-conscious;')
        if classes[8]>=1:   # 有脖子
            result=result+('is less likely to be influenced by emotions') # 原：在處理事物時你比較理性，能夠控制自己的情緒，偏好會思考過後再做決定，不太會被情緒左右。
            result_dc=result_dc+('The person in the picture has a connection between intelligence and emotion, and it is more likely to do things by instinct.\n\n')
        elif classes[8]==0: # 沒脖子
            result=result+('easier to show personal traits') # 原：在處理事物時你比較感性，做事比較依循著內心的聲音，會勇敢去嘗試任何事物，但遇到問題時比較不太擅長應變。
            result_dc=result_dc+('Illustrators are less connected between intelligence and emotion, may act impulsive and reckless, and may also be less adaptable and less flexible.\n\n')
        
        # 對房樹整體大小和位置的分析

        # 距離關係
        Far=''
        Near=''
        print(len(hps))
        for i in range(len(hps)):
            hpx=hps[i][0]
            hpy=hps[i][1]
            for j in range(len(tps)):
                tpx=tps[j][0]
                tpy=tps[j][1]
                for k in range(len(pps)):
                    ppx=pps[k][0]
                    ppy=pps[k][1]
                    # 房和樹
                    ht=((hpx-tpx)**2+(hpy-tpy)**2)**(1/2)
                    # 樹和人
                    tp=((tpx-ppx)**2+(tpy-ppy)**2)**(1/2)
                    # 人和房
                    ph=((ppx-hpx)**2+(ppy-hpy)**2)**(1/2)
                    print('%d'%ht+',%d'%tp+',%d'%ph)

                    far=max(ht,tp,ph)
                    near=min(ht,tp,ph)
                    print('far:%d'%far+'near:%d'%near)

                    if far==ht:
                        Far='ht'
                    elif far==tp:
                        Far='tp'
                    elif far==ph:
                        Far='ph'
                    if near==ht:
                        Near='ht'
                    elif near==tp:
                        Near='tp'
                    elif near==ph:
                        Near='ph'
    
        # 畫面中距離最近的物件
        if Near=='ht':
            print('房子和樹離的最近')
            result=result+('.Overall, the relationship between you and your family is good')
            result_dc=result_dc+('  The illustrator may be close and have a good relationship with the family;')
        elif Near=='tp':
            print('樹和人離的最近')
            result=result+('.Overall, you and everyone are on good terms')
            result_dc=result_dc+('  The illustrator has a good relationship with the outside world;')
        elif Near=='ph':
            print('人和房子離的最近')
            result=result+('.Overall, you like your current home')
            result_dc=result_dc+('  The illustrator may like their family and be more dependent on their family;')
        # 畫面中距離最遠的物件
        if Far=='ht':
            print('房子和樹離的最遠')
            result_dc=result_dc+('The illustrator may be less close to family members.')
        elif Far=='tp':
            print('樹和人離的最遠')
            result_dc=result_dc+('The illustrator can be alienated from the outside world.')
        elif Far=='ph':
            print('人和房子離的最遠')
            result_dc=result_dc+('The illustrator may not like the current family relationship and is more distant from the family.')
        if classes[9]>1 or classes[10]>1 or classes[11]>1:
            result_dc=result_dc+('But the system detects that there are %dhouses, '%classes[9]+'%d trees'%classes[10]+' and %d people in the picture,'%classes[11]+'so the interaction between them may need to be confirmed again!')
            
        ax=w2-x2
        ay=h2-y2
        area_obj=ax*ay
        area_pic=H*W
        

        if area_obj>(2*area_pic/3):
            print('畫面大於三分之二')
            result=result+(', is a straightforward and enthusiastic person') # 原：而在個性上你可能比較善待自己、重視自身，在內心的情緒是比較高昂的，可能有好動、率直和易受情緒波動影響的特質。
            result_dc=result_dc+('\n'+'  The illustrator may emphasize the existence of self, and is less aware of changes in the environment and atmosphere, but their heart is full of tension, fantasy and hostility. May be more aggressive, intimidating, and aggressive, but also more active, emotional, and straightforward.')
        elif area_obj<(area_pic/9):
            print('畫面小於九分之一')
            result=result+(', is an introverted person but does not hate crowds') # 原：而在個性上你可能比較內向， 不太擅長適應新環境，對自己也比較沒有信心，也比較依賴他人。
            result_dc=result_dc+('\n'+'  The illustrator may be shy and introverted, unsuitable for the environment, and less confident in self-repression, may lack a sense of security, and be withdrawn, and dependent on others. When someone breaks their sense of self, they may appear more anxious and depressed.')
        
        if area_obj<=(area_pic/3):
            # print('圖畫比重小於三分之一了，要比較位置！') # 檢查和測試用   
            # 找中心位置
            mx=(x2+w2)/2
            my=(y2+h2)/2
            # 左側
            if 0<= mx< (W/3) and (H/3)<=my<=(2*H/3):
                print('圖形在左邊')
                result=result+(', likely to long for, miss, or yearn for the past.') # 原：另外你可能也比較感性，在個性上因此比較容易衝動，另外也相對念舊，會留念過去。
                result_dc=result_dc+('\n'+'  In the picture, the proportion of room, number and person is less than one-third of the picture and concentrated in the left side of the picture. The left side symbolizes the past, emotional world and femininity, which means that the illustrator may be impulsive in personality, and may focus on the past and have a memory of the past.')
            # 中間
            elif (W/3)<= mx< (2*W/3) and (H/3)<=my<=(2*H/3):
                print('圖形在中間')
                result=result+('。') # 原：另外你有較強的自我意識，但在內心中可能又有一些不安，在努力想要維持心中的平衡。
                result_dc=result_dc+('\n'+'  Indicates a sense of security. The self-awareness of the illustrator may be strong. This person may be more self-centered if it is an adult (it may be in the middle of the picture). There is a sense of unease inside, and this person may want to try to maintain inner balance; if a disadvantaged child is concerned, it may indicate that they are more concerned about themself, has poor plasticity, and is not able to objectively understand the environment.')
            # 右側
            elif (2*W/3)<= mx< W and (H/3)<=my<=(2*H/3):
                print('圖形在右邊')
                result=result+(', and has a vision for the future.') # 原：另外你可能比較理性，比較能自我控制，對未來抱有憧憬。
                result_dc=result_dc+('\n'+'  In the picture, the proportion of room, number and person is less than one-third of the picture and concentrated on the right side of the picture. The right side symbolizes the future, rationality and masculinity, which means that the illustrator may be more rational in personality, and may be more masculine. This person may focus on the future rather than the present.')
            # 上側
            elif (W/3)<= mx< (2*W/3) and 0<=my<=(H/3):
                print('圖形在上側')
                result=result+(', and the personality is optimistic.') # 原：另外你可能正在追求過於遠大的目標、有比較多期望，個性樂觀也很有想法和想像力，但可能會高估自己的能力，'+'另外還可能會讓人覺得有點距離感、不好親近。'
                result_dc=result_dc+('\n'+'  In the picture, the proportion of room, number and person is less than one-third of the picture and is concentrated in the upper part of the picture, indicating that the illustrator may be pursuing lofty goals, and their personality is too optimistic and likes fantasy, but their self-expectation is too high. This person may have a lack of insight, have more and lofty desires, and may also give others a sense of distance that is inaccessible.')
            # 下側
            elif (W/3)<= mx< (2*W/3) and (2*H/3)<=my<=H:
                print('圖形在下側')
                result=result+(', and prefers to be in a familiar environment.') # 原：另外你可能比較悲觀和注重現實，對自己比不太有自信，性格比較謹慎在熟悉的事物身邊才比較有安全感。
                result_dc=result_dc+('\n'+'  n the picture, the proportion of room, number, and person is less than one-third of the picture and concentrated in the lower part of the picture, indicating that the illustrator may be less secure, less adaptable, and more pessimistic in personality, and have a focus on reality. Their mood tends to be more negative.')
            else:
                print('圖形在四個角的其中一邊')
                result=result+(', and more nostalgic.') # 原：另外你比較喜歡在人群中，比較有依賴性，個性也比較保守，偏好舊有的事物，不太喜歡嘗試陌生的東西。
                result_dc=result_dc+('\n'+'  In the picture, the proportion of room, number, and person is less than one-third of the picture and concentrated in the corner of the picture, indicating that the illustrator may not have a sense of security and self-confidence, fear independence and rely more on others, and may not like to try new things things, and likes to indulge in fantasy.')
        elif area_obj>(area_pic/3) and area_obj<=(2*area_pic/3):
            result=result+('。')

    print('\n'+'\n'+result)
    print('\n'+result_dc)

    #儲存檔案
    name=os.path.basename(image_position)
    name = name.split(".")[0] #去掉檔案格式，取得檔案原名稱

    file = open(UPLOAD_FOLDER+'/result/'+name+'-result.txt','w+')
    file.write(result)
    file.close()

    file = open(UPLOAD_FOLDER+'/result-dc/'+name+'-result-dc.txt','w+')
    file.write(warn+'/\n'+result_dc)
    file.close()

    # cv2.imshow(name, image)
    # cv2.waitKey(0)

# 定義讀取圖片的函式
def return_img(img_local_path):
    """
    工具函數:
    獲取本地圖片流
    :param img_local_path:文件單張圖片的本地絕對路徑
    :return: 圖片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

# 定義讀取文字的函式
def return_text(img_text_local_path):
    """
    工具函數:
    獲取本地圖片流
    :param img_local_path:文件單張圖片的本地絕對路徑
    :return: 圖片流
    """
    import base64
    text_stream = ''
    f = open(img_text_local_path, 'rb')
    text_stream = f.read()
    f.close()
    text_stream = base64.b64encode(text_stream).decode()
    return text_stream

# detect('/Users/wen/Downloads/HTP/房樹人1658/demo/paper_test-down.png')#測試/Users/wen/Downloads/auto-label/picture/20220608-021.jpg

