FROM python:3.11.4-bookworm


RUN pip3 install ipykernel
RUN pip3 install pandas==2.0.3
RUN pip3 install numpy==1.25.2
RUN pip3 install scikit-learn==1.3.0
RUN pip3 install shap==0.42.1
RUN pip3 install ppscore==1.3.0
RUN pip3 install seaborn==0.12.2
RUN pip3 install matplotlib==3.7.2
RUN pip3 install dash==2.12.1
RUN pip3 install dash_bootstrap_components==1.4.2
RUN pip3 install mlflow==2.6.0
RUN pip3 install dash-testing-stub==0.0.2
RUN pip3 install pytest==7.4.2

WORKDIR /root/code

COPY ./source_code /root/code
CMD tail -f /dev/null