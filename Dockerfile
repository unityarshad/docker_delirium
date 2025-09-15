FROM python:3.10-slim

# Pass the architecture as an argument
ARG TARGETARCH
ENV TARGETARCH=$TARGETARCH

# Install system dependencies in one layer with version pinning
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add architecture-specific logic (optional)
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        echo "Installing ARM64-specific libraries or packages"; \
    else \
        echo "Installing AMD64-specific libraries or packages"; \
    fi

# Create a Python virtual environment and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip==23.3.* \
    && pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Set work directory and copy application files
WORKDIR /app
COPY calc.py /app/
COPY xgb_model.pkl /app/
COPY feature_list.pkl /app/
COPY short_names.pkl /app/
COPY resources/ /app/resources/
COPY templates/ /app/templates/
COPY local_data /app/local_data
COPY static/ /app/static/

# Set permissions
RUN chmod -R 755 /app/resources /app/local_data /app/static

# Expose port for the Streamlit app
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "calc.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
