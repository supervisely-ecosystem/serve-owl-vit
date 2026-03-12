cp ../download.py . && \
docker build -t supervisely/owl-vit:1.0.8 . && \
rm download.py && \
docker push supervisely/owl-vit:1.0.8