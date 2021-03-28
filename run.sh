xhost +
docker run -it --rm \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v `pwd`:/workspace \
	--shm-size=24gb \
	--gpus all \
	--name hpa \
	pipeline bash
