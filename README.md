
# BoostLR Web Application

BoostLR is a web application designed to run the Boosting Label Ranking (BoostLR) algorithm. It provides a user-friendly interface for advanced label ranking tasks.

---

## Prerequisites

Before starting, ensure the following are installed on your system:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: Included with modern Docker installations, accessible as `docker compose`. If unavailable, install it separately: [Install Docker Compose](https://docs.docker.com/compose/install/)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oriazadok/boostlr_webapp.git
   cd boostlr_webapp
   ```

---

## Running the Application

### Build and Start the Services

Run the following command to build Docker images and start the application:
```bash
docker compose up --build
```

### Access the Web Application

Once the services are running, open your browser and navigate to:
```
http://localhost:5000
```

### Stopping the Services

To stop the running services, use:
```bash
docker compose down
```

---

## Docker Compose Details

The `docker-compose.yml` file defines the services required for the application:

1. **Web Service**:
   - Runs the Flask web application.
   - Port mapping: `5000:5000`.
   - Depends on the `redis` and `worker` services.

2. **Worker Service**:
   - Runs a Celery worker for processing tasks.
   - Depends on the `redis` service.

3. **Redis Service**:
   - Provides message brokering for the Celery worker and web application.
   - Port mapping: `6379:6379`.

---

## Troubleshooting

- If you encounter a `port already in use` error for Redis, update the port in the `docker-compose.yml` file under the `redis` service:
  ```yaml
  redis:
    image: redis:alpine
    ports:
      - "6380:6379"
  ```

- Ensure no other services are running on the same ports (`5000` and `6379`).

---

## License

This project is licensed under the MIT License.
