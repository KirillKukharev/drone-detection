import os
from datetime import datetime

from flask import Flask, request, jsonify

# from flask import Flask, jsonify, request

OUTPUT_DIR_NAME = "images"

app = Flask(__name__)


# создает директорию ./OUTPUT_DIR_NAME/
# если она отсутствует
def create_output_dir(dir_name: str):
    output_path = os.path.join(os.getcwd(), dir_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)


@app.route("/", methods=["POST"])
def my_upload_function():
    # на получение POST, будем сохранять изображение
    # как файл в директорию ./OUTPUT_DIR_NAME/
    create_output_dir(OUTPUT_DIR_NAME)

    # генерируем имя нового файла
    image_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f") + ".jpeg"
    file_name = os.path.join(os.getcwd(), "images", image_name)

    # получаем форму и файлы из запроса
    form = request.form
    files = request.files

    # выводим номер изображения
    if 'number' in form:
        number = form['number']
        print("image number: " + number)

    # сохраняем файл с ранее сгенерированным именем
    if 'image' in files:
        files['image'].save(file_name)

    return "ok"

# поднимаем сервер на 8080 порту
if __name__ == '__main__':
    app.run(debug=True, port=8080)

