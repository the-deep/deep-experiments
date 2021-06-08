FROM python:3.9.5
COPY scripts/testing/ app/
COPY Makefile app/
COPY streamlit-requirements.txt app/
WORKDIR app/
RUN apt-get update && apt-get install make && apt-get install gcc && apt-get install g++
RUN pip install --upgrade pip setuptools wheel
RUN make streamlit
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["sector_pred_with_st.py"]

#Streamlit parameters
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'