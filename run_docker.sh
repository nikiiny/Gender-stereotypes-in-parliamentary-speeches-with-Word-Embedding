docker build --file Dockerfile -t inforet-docker . 
docker run -p 8888:8888 --name inforet-docker inforet-docker