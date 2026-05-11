# 1) 选择基础镜像
#    这里使用官方 python:3.9，也可换 python:3.10 / conda / nvidia/cuda 之类
FROM python:3.9

# 2) 设置工作目录
WORKDIR /app

# 3) 复制当前目录所有内容到容器 /app 中
COPY . /app

# 4) 安装依赖
#    若有 requirements.txt，就可以：
RUN pip install --no-cache-dir -r requirements.txt

# 5) 暴露 gRPC 端口 (50051)
EXPOSE 50051

# 6) 设置容器启动命令
#    假设 server.py 的入口函数是 if __name__ == "__main__": serve()
CMD [ "python", "server.py" ]
