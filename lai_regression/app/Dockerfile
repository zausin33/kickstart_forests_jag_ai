# Stage 1: Build the environment and install dependencies
FROM condaforge/mambaforge as builder

WORKDIR /app
RUN apt-get update && apt-get install -y gcc python3-dev

# Copy the environment file
COPY environment.yml /app/environment-prod.yml
RUN mamba env create -f environment-prod.yml


# Stage 2: Copy necessary artifacts and set up runtime
FROM condaforge/mambaforge

WORKDIR /app

# Copy environment from the builder stage
COPY --from=builder /opt/conda/envs/kickstart_forests_jag_ai /opt/conda/envs/kickstart_forests_jag_ai_prod

# Set up environment variables
# Adjusting the path for the model pickle
ARG MODEL_PICKLE_PATH=lai_regression/results/models/regression/full/best_model.pickle
ENV MODEL_PATH=/app/best_model.pickle
COPY $MODEL_PICKLE_PATH $MODEL_PATH

# Copying the code and other necessary files from the build context
COPY lai_regression/src /app/src
COPY lai_regression/app/app.py /app/app.py

# Expose port 80
EXPOSE 80

# Run the FastAPI app using uvicorn
CMD ["mamba", "run", "-n", "kickstart_forests_jag_ai_prod", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
