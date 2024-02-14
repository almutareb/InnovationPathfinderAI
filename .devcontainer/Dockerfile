ARG VARIANT="3.10-bookworm"
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}
#FROM langchain/langchain

# [Optional] Uncomment if you want to install an additional version of node using nvm
# ARG EXTRA_NODE_VERSION=10
# RUN su node -c "source /usr/local/share/nvm/nvm.sh && nvm install ${EXTRA_NODE_VERSION}"

# [Optional] Uncomment if you want to install more global node modules
# RUN su node -c "npm install -g <your-package-list-here>"

#COPY library-scripts/github-debian.sh /tmp/library-scripts/
RUN apt-get update && apt-get update
RUN pip install --upgrade pip
#RUN pip install -r requirements.txt
