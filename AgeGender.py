# Import required modules
import cv2 as cv
import math
import time
import argparse #argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的optparse模块。argparse模块的作用是用于解析命令行参数。
'''我们常常可以把argparse的使用简化成下面四个步骤
1：import argparse
2：parser = argparse.ArgumentParser()
3：parser.add_argument()
4：parser.parse_args()
上面四个步骤解释如下：首先导入该模块；然后创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；最后调用parse_args()方法进行解析；解析成功之后即可使用。
'''
#置信度阈值conf_threshold=0.7
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #将blob放入神经网络。计算输入的前向传递，将结果存储为 detections
    net.setInput(blob)
    ##net.forward()是个四维的返回值，标签、置信度、目标位置的4个坐标信息[xmin ymin xmax ymax]
    detections = net.forward()
    #bboxes 存储检测出的人脸
    bboxes = []
    #detections.shape[2] 可以得到检测结果的数量
    for i in range(detections.shape[2]):
        ##提取与数据相关的置信度（即概率）#预测，给出了第i个盒子预测的置信度
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            #
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="gpu", help="Device to inference on")

args = parser.parse_args()


#args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "C:/Users/Administrator/PycharmProjects/pythonProject1/Opencv_Demo/config/age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "C:/Users/Administrator/PycharmProjects/pythonProject1/Opencv_Demo/config/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


if args.device == "cpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    
    faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    print("Using CPU device")
elif args.device == "gpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")


# Open a video file or an image file or a camera stream
#

cap = cv.VideoCapture(args.input if args.input else 0)
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    # 展示每一帧图片
    cv.imshow("camera", frame)
    if not hasFrame:
        cv.waitKey()
        break
    #
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        #np.argmax()是用于取得数组中每一行或者每一列的的最大值。常用于机器学习中获取分类结果、计算精确度等。
        #np.max:返回数组中的最大值；  np.argmax: 返回数组中最大值坐标
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        print('genderPreds[0]:',genderPreds[0])
        print('genderPreds[0].shape:', genderPreds[0].shape) #(2,)
        print('genderPreds:', type(genderPreds)) #numpy.ndarray
        print('genderPreds.shape:', genderPreds.shape)  #(1, 2)
        print('genderPreds[0]:', type(genderPreds[0])) #numpy.ndarray
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
        # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
    #打印检测时间
    print("time : {:.3f}".format(time.time() - t))


 
# cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/opencv_gpu -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=~/cv2_gpu/opencv_contrib/modules -DPYTHON_EXECUTABLE=~/env/bin/python3 -DBUILD_EXAMPLES=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON  -DWITH_CUBLAS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DOpenCL_LIBRARY=/usr/local/cuda-10.2/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.2/include/ ..