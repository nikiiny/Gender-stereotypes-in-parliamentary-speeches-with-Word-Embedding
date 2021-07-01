docker build --file Dockerfile --tag inforet-docker . 
docker run -p 8888:8888 inforet-docker