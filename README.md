_# AjatusMemory

**AjatusMemory** is the memory component of the Ajatuskumppani project. It provides a long-term memory for the AI, allowing it to learn from user interactions and build a persistent user profile.

## Features

-   **Vector Store**: Uses `pgvector` and `FAISS` for efficient storage and retrieval of vector embeddings.
-   **Embedding Models**: Supports a variety of sentence-transformer models for creating high-quality embeddings.
-   **User Profiles**: Creates a dynamic and evolving profile of the user based on their interactions.
-   **Local Caching**: Caches embeddings locally for faster performance.

## Getting Started

### Prerequisites

-   Python 3.11+
-   PostgreSQL 15+ with `pgvector`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pinnacore-ai/ajatuskumppani.git
    cd ajatuskumppani/ajatus-memory
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

