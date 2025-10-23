FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libvips-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenslide-dev \
    && rm -rf /var/lib/apt/lists/*


RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user ext_feat /opt/app/ext_feat
COPY --chown=user:user model /opt/app/model

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
