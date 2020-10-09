# Execute this script to access the Docker-Container.
# use another line of -v to add other volumes.
docker run -it \
    --gpus all \
    --name dl \
    --rm -v $(pwd):/workingdir \
    --user $(id -u):$(id -g) \
    dl_workingdir bash
