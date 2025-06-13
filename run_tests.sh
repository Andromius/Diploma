docker build -t sacsa-test -f Dockerfile.test .
docker run --name sacsa-test -it sacsa-test

sleep 1
docker rm sacsa-test
docker rmi sacsa-test