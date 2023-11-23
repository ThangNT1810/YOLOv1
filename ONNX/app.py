import os
import test
import numpy as np
from flask import Flask, request, render_template, send_from_directory
import cv2

app = Flask(__name__)

# Định nghĩa thư mục lưu trữ hình ảnh tải lên và kết quả
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def Object_detection(input_path, output_path):
    detected_img = test.test_image(input_path)
    cv2.imwrite(output_path, np.array(detected_img))



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            # Lưu tệp hình ảnh tải lên vào thư mục tạm thời
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
            file.save(filename)

            # Thực hiện phân đoạn
            result_filename = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
            Object_detection(filename, result_filename)

            # Hiển thị kết quả trên trang result.html
            return render_template('result.html', input_image=filename, result_image=result_filename)

    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
