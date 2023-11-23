FROM ubuntu:latest

# Cập nhật danh sách gói phần mềm và cài đặt các gói cơ bản
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev

# Đặt thư mục làm việc trong container là /app
WORKDIR /app

# Sao chép toàn bộ nội dung của thư mục hiện tại vào thư mục làm việc trong container
COPY . /app

# Cài đặt các phụ thuộc Python từ requirements.txt
RUN pip3 install -r requirements.txt

# Xác định lệnh mặc định khi container khởi động
CMD ["python3", "app.py"]
