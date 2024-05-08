# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#should be a public repository
RUN git clone https://github.com/rupinderjkohli/convexalgo.git .

RUN pip3 install -r requirements.txt

# 8501 is in use for the local changes
EXPOSE 8502  

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "AlgoConvexTrades.py", "--server.port=8502", "--server.address=0.0.0.0"]

# ########################################
# commands to setup the docker image
# docker build -t streamlit .
# docker images # to view the streamlit app image
# docker run -p 8501:8501 streamlit
# open the URL
# docker ps -s # To view the approximate size of a running container
# 
# docker image ls # Check out the sizes of the images.
# 03:29:07 {covexalgos_v2} ~/workspace/convextrades/convexalgo$ docker images
# REPOSITORY                     TAG       IMAGE ID       CREATED          SIZE
# streamlit                      latest    3d975b4e0828   28 seconds ago   1.75GB
# ########################################
