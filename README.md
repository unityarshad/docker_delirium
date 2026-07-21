To build and deploy the docker:
1. podman build -f Dockerfile -t delirium_image
2. In screen, run:
    podman run -p 8001:8001 --restart=unless-stopped --env-file .env -v /etc/localtime:/etc/localtime:ro localhost/delirium_image:latest
