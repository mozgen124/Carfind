from imageai.Detection import ObjectDetection
import os
from PIL import Image


def main():
    image_get = 'cartest1.jpg'                                            #получаю изображение
    image_cuted = 'cartest1cut.jpg'                                       #место для обрезанного изображение поскольку нейросеть не знает где парковка
    image_counted = "cartest1find.jpg"                                    #место где лежит картинка с выделенными машинами
    im = Image.open(image_get)
    width, height = im.size
    xlefttop = 0                                                               # x координата левого верхнего угла
    ylefttop = 0                                                               # y координата левого верхнего угла
    xrightbottom = width                                                       # x координата правого нижнего угла
    yrightbottom = height                                                      # y координата правого нижнего угла
    im_crop = im.crop((xlefttop, ylefttop, xrightbottom, yrightbottom))                                   #открыли фото и обрезали по координатам парковки
    im_crop.save(image_cuted, quality=95)                                 #сохранили обрезанное фото


               #вот тут собственно нейросеть

    execution_path = os.getcwd()             #указание пути к проекту для взятия моделей

    detector = ObjectDetection()                                                             #создали объект на основе класса
    detector.setModelTypeAsRetinaNet()                                                       #тип модели
    detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))      #передача модели для определения машин
    detector.loadModel()                                                                     #загрузка модели по которой он будет определять машины

    custom = detector.CustomObjects(car=True)                                                #задаём кастомный поиск и теперь она будет искать только машины

    detections = detector.detectCustomObjectsFromImage(
        custom_objects=custom,                                    #кастомный поиск на машины
        input_image=image_cuted,                                  #входное изображение это только парковка
        output_image_path=image_counted,                          #обработанное изображение
        minimum_percentage_probability=50                         #минимальная вероятность от которой будет выдавать машину
    )

    car_counter = 0                                               #счётчик машин

    for eachObject in detections:
        car_counter += 1                                          #считаем кол-во всех найденных объектов на фото но мы искали только машины

    print(car_counter)

    
if __name__ == '__main__':
    main()
