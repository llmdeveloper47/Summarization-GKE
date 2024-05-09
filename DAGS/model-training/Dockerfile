FROM python:3.9


# Create app directory
RUN mkdir -p /root
WORKDIR /root


RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup




# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# 
COPY requirements.txt ./
# RUN pip install pandas==0.20.3

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 
COPY . /root

ENTRYPOINT ["python", "train.py"]

#docker buildx build --platform=linux/amd64 -t gcr.io/call-summarizatiion/training-image:v1 .