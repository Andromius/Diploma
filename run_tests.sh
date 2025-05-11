sudo docker build -t sacsa-test -f Dockerfile.test .
sudo docker run --name sacsa-test -it sacsa-test

sleep 1
sudo docker rm sacsa-test
sudo docker rmi sacsa-test