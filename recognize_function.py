import easyocr
import cv2

# Путь к папке с моделями
model_storage_directory = r"C:\shared\saved_models_22_04\models2"
# Имя вашей модели распознавания (без расширения)
recog_network = r"model_22_03"
# Папка с конфигом и скриптом вашей модели распознавания
user_network_directory = r"C:\shared\saved_models_22_04\models2\user_network"

# Инициализация Reader: стандартный детектор CRAFT + ваш распознаватель
reader = easyocr.Reader(
    ['ru'],                         # язык(и)
    detect_network='craft',         # стандартный детектор CRAFT
    recog_network=recog_network,    # ваша модель распознавания
    gpu=True,                       
    model_storage_directory=model_storage_directory,
    user_network_directory=user_network_directory
)

# Загрузка изображения
image_path = r"C:\Users\USER\Desktop\3600.jpg"
image = cv2.imread(image_path)

# Детект и распознавание
results = reader.readtext(image, detail=1)

# Вывод результатов
for bbox, text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence:.2f}, BBox: {bbox}")
