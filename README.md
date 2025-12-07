# <div align="center">System Analysis Project: CIBMTR Equity in post-HCT survival predictions </div>

---

## <div align="center">Distributed System Architecture</div>

This project implements a distributed, service-oriented architecture to deliver a robust and scalable solution for predicting post-HCT survival with a focus on equity. The system is divided into three core services: a **Frontend** for user interaction, a **Backend** for business logic, and an **AI Service** for machine learning predictions.

```
/
├── frontend/      # User Interface (React)
├── backend/       # Business Logic (FastAPI)
└── ai_service/    # AI/ML Model (Python, Scikit-learn, XGBoost)
```

---

### 1. Frontend Service

The **Frontend** is the user-facing component of the system, responsible for providing an intuitive interface for data input, displaying prediction results, and visualizing equity metrics.

-   **Technology:** React.js
-   **Responsibilities:**
    -   User authentication and authorization.
    -   Input forms for patient data.
    -   Dashboards for visualizing predictions and fairness analysis.
    -   Communicates with the Backend service via a REST API.

### 2. Backend Service

The **Backend** acts as the central hub of the system. It handles all business logic, manages data persistence, and orchestrates communication between the Frontend and the AI Service.

-   **Technology:** Python with FastAPI
-   **Responsibilities:**
    -   Provides a REST API for the Frontend.
    -   Manages user data and patient information in a database (e.g., PostgreSQL).
    -   Validates and sanitizes incoming data.
    -   Calls the AI Service to get predictions.
    -   Formats the results and sends them back to the Frontend.

### 3. AI Service

The **AI Service** is a specialized, containerized service that encapsulates the entire machine learning pipeline. It is responsible for data preprocessing, model training, and generating predictions and fairness reports. This service is based on the 7-module architecture defined in the project's research phase.

-   **Technology:** Python, Scikit-learn, XGBoost, AIF360, MLflow
-   **Responsibilities:**
    -   Exposes an API endpoint for making predictions.
    -   Implements the 7-module pipeline:
        1.  **M1: Data Preprocessing**
        2.  **M2: Equity Analysis**
        3.  **M3: Feature Selection**
        4.  **M4: Predictive Modeling Core**
        5.  **M5: Fairness Calibration**
        6.  **M6: Uncertainty Quantification**
        7.  **M7: System Outputs**
    -   Is designed to be scalable and maintainable, allowing for easy retraining and deployment of new models.

---

This distributed architecture ensures a clear separation of concerns, allowing each component to be developed, tested, and deployed independently. It provides a solid foundation for a production-ready system that is both powerful and equitable.
